from __future__ import print_function

import math
import os

import numpy as np
import pandas

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
import sklearn.metrics

import torch
import torch.utils.data
import torchvision

from data.simulation import simulation_cox_weibull, formatted_data_simu, saveDataCSV
from data.EVT_dataloader import EVTDataset, EVTDataset_dic,ImbalancedDatasetSampler, callback_get_label
from utils.distributions import mixed_loglikeli, loglog_function, sample_mixedGPD, log_sum_exp
from utils.preprocessing import loadDataDict, flatten_nested, datadicTimeCut_delcensor
from data.sleep_data import generate_data

# from networks.VIEVT import IAF, Decoder, Nu, log_score_marginal
# from networks.VIEVT import testing_VIEVT, pred_avg_risk
from networks.VIEVT_outHz import IAF, Decoder, Nu, log_score_marginal
from networks.VIEVT_outHz import testing_VIEVT, pred_avg_risk

from utils.metrics import binary_cross_entropy,  view_distribution_z_e_hz, view_z_e, view_z_box, view_z_dist
from utils.metrics import boostrappingCI


from pathlib import Path

from utils.preprocessing import loadDataDict, flatten_nested, datadicTimeCut_delcensor

# Load SLEEP dataset 
df=generate_data()
train_o, valid_o, test_o = df['train'], df['test'], df['valid']
del df

df={'x': np.concatenate([train_o['x'], valid_o['x'], test_o['x']],axis=0),\
    'e': np.concatenate([train_o['e'], valid_o['e'], test_o['e']],axis=0),\
    't': np.concatenate([train_o['t'], valid_o['t'], test_o['t']],axis=0)}

n_samples, ncov = df['x'].shape

# # cut as a whole
# # cut as a whole
# data_name = 'er05'
# df = datadicTimeCut(df, time_cut=600)
# seed = 1234
# lambda_ = [1.0, 1e-3, 1e-5]

data_name = 'er01'
df = datadicTimeCut_delcensor(df, time_cut=150)
seed=1111
lambda_ = [1.0, 1e-4, 1e-6]


np.random.seed(seed)
perm_idx = np.random.permutation(n_samples)
train_idx = perm_idx[0:int(3*n_samples/6)]
valid_idx = perm_idx[int(3*n_samples/6):int(4*n_samples/6)]
test_idx = perm_idx[int(4*n_samples/6):n_samples]

train = formatted_data_simu(df['x'], df['t'], df['e'], train_idx)
test = formatted_data_simu(df['x'], df['t'], df['e'], test_idx)
valid = formatted_data_simu(df['x'], df['t'], df['e'], valid_idx)

np.mean(train['e']), np.mean(valid['e']), np.mean(test['e'])

del df, train_o, test_o, valid_o


result_path_root = './results/'
result_path = result_path_root+"SLEEP"+'/'+data_name
Path(result_path).mkdir(parents=True, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)
# device = torch.device('cpu')


model_path = result_path+"/saved_models"
Path(model_path).mkdir(parents=True, exist_ok=True)
plot_path = result_path+"/plots"
Path(plot_path).mkdir(parents=True, exist_ok=True)


event_rate = np.mean(train['e'])
ncov = train['x'].shape[1]
########## Hyper-parameters##############
########## Hyper-parameters##############
# set hyperparameters
model_name = 'VIE'
z_dim = 4


hidden_layers=[32,32,32]
# eps_dim = np.int(ncov)
eps_dim = np.int(ncov)

input_size = ncov+eps_dim
unroll_steps = 5
nu_lambda=1.0
epochs = 500
batch_size = 200

flow_path = result_path+"/saved_models/"+model_name+'_flow'+".pt"

decoder_path = result_path+"/saved_models/"+model_name+'_decoder'+".pt"
nu_path = result_path+"/saved_models/"+model_name+'_nu'+".pt"

training = True

unroll_test = True

u_bound = np.max([0.99, 1-event_rate])
lower_bound = -5.0
N = 100


IAF_flow = IAF(input_size, z_dim=z_dim, h_dim=z_dim, hidden_layers=hidden_layers, nstep=5, device=device)
decoder = Decoder(z_dim=z_dim, hidden_layer_MNN=[32,32,32],loglogLink=True)
nu = Nu(z_dim=z_dim, ncov=ncov, hidden_layers=[32,32], marginal=True)

decoder.to(device)
IAF_flow.to(device)
nu.to(device)

# define optimizer
opt_flow = optim.Adam(IAF_flow.parameters(), lr=1e-4)
opt_dec = optim.Adam(decoder.parameters(), lr=1e-4)
opt_nu = optim.RMSprop( nu.parameters(), lr = 1e-3)

aggressive_flag = True
aggressive_nu = True

# splitting to training/validation/testing
cat_covariates = np.array([])
continuous_variables = np.setdiff1d(np.arange(ncov), cat_covariates)


# consider normaliztion of inputs
norm_mean = np.mean(train['x'][:,continuous_variables],axis=0)
norm_std = np.std(train['x'][:,continuous_variables],axis=0)

# delete variable with 0 std
continuous_variables = np.delete(continuous_variables, np.where(norm_std==0.0)[0])

norm_mean = np.nanmean(train['x'][:,continuous_variables],axis=0)
norm_std = np.nanstd(train['x'][:,continuous_variables],axis=0)



EVT_train = EVTDataset_dic(train,transform=True,norm_mean=norm_mean, norm_std=norm_std, continuous_variables=continuous_variables)
EVT_valid = EVTDataset_dic(valid,transform=True,norm_mean=norm_mean, norm_std=norm_std, continuous_variables=continuous_variables)




#
# train with imbalanced sampler
train_loader = DataLoader(EVT_train, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train, callback_get_label=callback_get_label))
# valid_loader = DataLoader(EVT_valid, batch_size=batch_size*10, sampler=ImbalancedDatasetSampler(valid, callback_get_label=callback_get_label))
# validation on the original scale
valid_loader = DataLoader(EVT_valid, batch_size=1000, shuffle=True)

del train
## define aggressive training
def agrressive_step():
    opt_flow.zero_grad()    
    opt_dec.zero_grad()
    
    best_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
    assert (best_z != best_z).any()== False 
    pred_risk_cur = decoder(best_z, N, lower_bound).float()
    BCE_loss = binary_cross_entropy(pred_risk_cur, \
                                    batched_e.detach().float(), sample_weight=batch_weight.float())

    z_nu, pz_nu, nanFlag = log_score_marginal(nu=nu, z=best_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                            xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                            p_ = u_bound, eps=1e-3, nu_lambda=nu_lambda,device=device, train_nu=False)
    
    # calculate KL(q(z|x)||p(z))
    
    likelihood_pz = mixed_loglikeli(best_z, IAF_flow.mu0, IAF_flow.logvar0, IAF_flow.xi_, IAF_flow.sigma_, u_bound)
    assert (likelihood_pz != likelihood_pz).any()== False 
    KL_cond = likelihood_qzx.sum() - likelihood_pz.sum()
    loss = lambda_[0]*BCE_loss + lambda_[1]*(z_nu - pz_nu) + lambda_[2]*KL_cond
    loss.backward()
    torch.nn.utils.clip_grad_norm_(IAF_flow.parameters(), 1e-6)
    opt_flow.step()
    
    return loss.item()


# training process

if __name__ == "__main__":
    if training:
        best_valid_loss = np.inf
        best_valid_recon_loss = np.inf
        best_valid_pos_loss = np.inf
        best_valid_auc = 0
        best_epoch = 0
        nanFlag = 0
        
        # save training process
        
        train_z_nu = []
        train_pz_nu = []
        train_KL = []
        train_BCE = []
        last_shrink = 0
#         model.train()
        for epoch in range(1, epochs + 1):
            if nanFlag == 1:
                break
                
    #         train(epoch)
    #         test(epoch)
            train_loss = 0
            valid_loss = 0
            valid_recon_loss = 0
            valid_pos_loss = 0
            pre_mi = 0
            
            improved_str = " "

            # detect errors
#             with torch.autograd.detect_anomaly():
            for batch_idx, batched_sample in enumerate(train_loader):
#                 print(batch_idx)
                if nanFlag == 1:
                    break
                IAF_flow.train()
                decoder.train()
                nu.train()
                
                batched_x =  batched_sample['x']
                batched_x = batched_x.to(device).view(-1, ncov)
                batched_e =  batched_sample['e'].to(device)
                batch_weight = batched_e.clone().detach().data*event_rate + (1-batched_e.clone().detach().data)*(1-event_rate)
                
                # add noise
                eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)                
                best_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
                
                try:
                    assert (best_z != best_z).any()== False
                    
                except AssertionError:
                    break
                # aim to update nu based on conditional q
                # update multiple times of the critic
                if aggressive_nu:
                    if epoch > 10:
                        aggressive_nu = False
                        print("STOP multiple learning of nu")
                    for iter_ in range(unroll_steps):
                    ## conditional posterior
                        # aim to update nu based on marginal q
                        z_nu, pz_nu, loss_nu, nanFlag = log_score_marginal(nu=nu, z=best_z, \
                                                                           mu=IAF_flow.mu0, logvar=IAF_flow.logvar0,\
                                                                           xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                                                                           p_ = u_bound, eps=1e-3, nu_lambda=nu_lambda,\
                                                                           device=device,train_nu=True, opt_nu=opt_nu)           


                        if ((1*torch.isnan(best_z)).sum() + (1*torch.isnan(pz_nu)).sum()  + (1*torch.isnan(z_nu)).sum()).item()>0:
                            print("NaN occured at critic training")
    #                         print(z_init)
                            print(IAF_flow.xi_, IAF_flow.sigma_, IAF_flow.mu0, IAF_flow.logvar0)
                            nanFlag = 1
                            break 
                else:
                    z_nu, pz_nu, loss_nu, nanFlag = log_score_marginal(nu=nu, z=best_z,\
                                                                       mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                                                       xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                                                                       p_ = u_bound, eps=1e-3, nu_lambda=nu_lambda,\
                                                                       device=device, train_nu=True, opt_nu=opt_nu)                                
    
                # update encoder and decoder's parameters
                
                if aggressive_flag:    
                    sub_iter = 0
                    while sub_iter  < 10:

                        sub_loss = agrressive_step()
    #                     print(sub_iter,sub_loss)
                        sub_iter += 1
                    
                    
                opt_dec.zero_grad()
                opt_flow.zero_grad()
                
                BCE_loss = binary_cross_entropy(decoder(best_z, N, lower_bound).float(), \
                                                batched_e.detach().float(), sample_weight=batch_weight.float())

                z_nu, pz_nu, nanFlag = log_score_marginal(nu=nu, z=best_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                        xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                                        p_ = u_bound, eps=1e-3, nu_lambda=nu_lambda,device=device, train_nu=False)
                
                likelihood_pz = mixed_loglikeli(best_z, IAF_flow.mu0, IAF_flow.logvar0, IAF_flow.xi_, IAF_flow.sigma_, u_bound)
                KL_cond = likelihood_qzx.sum() - likelihood_pz.sum()
#                 print(likelihood_qzx, likelihood_pz.sum())
                loss = lambda_[0]*BCE_loss + lambda_[1]*(z_nu - pz_nu) + lambda_[2]*KL_cond

                loss.backward()
                
    
                train_z_nu.append(z_nu.item())
                train_pz_nu.append(pz_nu.item())
                train_BCE.append(BCE_loss.item())
                train_KL.append(KL_cond.item())
                
                
                train_loss += loss.item()
                if not aggressive_flag:
                    torch.nn.utils.clip_grad_norm_(IAF_flow.parameters(), 1e-6)
                    opt_flow.step()

                opt_dec.step()                
                
                            
            
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                          epoch, train_loss))

            if nanFlag == 1:
                    break
            # check performance on validation dataset
#             with torch.no_grad():
            if nanFlag == 0:
                IAF_flow.eval()
                decoder.eval()
                nu.eval()
                for i, batched_sample in enumerate(valid_loader):
                    batched_x =  batched_sample['x']
                    batched_x = batched_x.to(device).view(-1, ncov)
                    batched_e =  batched_sample['e'].to(device)
                    # add noise
                    eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)
                    batch_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())

                    if aggressive_flag:
                        cur_mi = likelihood_qzx.sum() - (log_sum_exp(likelihood_qzx)).sum()
                        if cur_mi - pre_mi < 0:
                            aggressive_flag = False
                            print("STOP aggressive learning")
                        cur_mi = pre_mi
                        
                    
#                     pred_risk_batch = decoder(batch_z, N, lower_bound)
                    pred_risk_batch, likelihood_qzx= pred_avg_risk(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=1)

                    valid_recon_, pos_recon_ = binary_cross_entropy(pred_risk_batch.float(), \
                                             batched_e.detach().float(), sample_weight=None, pos_acc=True)
        
                    # based on marginal q
                    z_nu, pz_nu,nanFlag = log_score_marginal(nu=nu, z=batch_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                            xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                                            p_ = u_bound, eps=1e-3, device=device, train_nu=False)

                    # based on conditional q
                    likelihood_pz = mixed_loglikeli(batch_z, IAF_flow.mu0, IAF_flow.logvar0, IAF_flow.xi_, IAF_flow.sigma_, u_bound)
                    KL_cond = likelihood_qzx.sum()  - likelihood_pz.sum()    
            
                    valid_loss_ = valid_recon_ + z_nu - pz_nu + KL_cond

                    # calculating AUC
                    pred_risk = pred_risk_batch.cpu().detach().squeeze().numpy()
                    nonnan_idx = np.where(np.isnan(pred_risk)==False)[0]
                    pred_risk = pred_risk[nonnan_idx]
                    valid_auc_ = sklearn.metrics.roc_auc_score(batched_sample['e'][nonnan_idx,:].cpu().squeeze().numpy(),\
                                                               pred_risk).item()
#                     # calculating F1 score                    
#                     valid_F1 = F1_score(batched_sample['e'].cpu().squeeze().numpy(),\
#                                         pred_risk_batch.cpu().detach().squeeze().numpy(), beta=1.0)
                    
                    valid_loss = valid_loss + valid_loss_.item()
                    valid_recon_loss = valid_recon_loss + valid_recon_.item()
                    valid_pos_loss = valid_pos_loss + pos_recon_.item()
                    break

                
                # only save non-nan models
                if np.isnan(valid_recon_loss) == False:
                    save_model = 0
                    if (valid_recon_loss < best_valid_recon_loss) or (valid_pos_loss < best_valid_pos_loss) or (valid_auc_ > best_valid_auc):

                        if (valid_recon_loss < best_valid_recon_loss):
    #                         best_valid_recon_loss = valid_recon_loss
        #                     torch.save(model.state_dict(), model_path)

                            save_model += 1
                        if (valid_pos_loss < best_valid_pos_loss):
    #                         best_valid_pos_loss = valid_pos_loss
                            save_model += 1
                        if (valid_auc_ > best_valid_auc):
    #                         best_valid_auc = valid_auc_
                            save_model += 1


                        # save current model

                    if save_model > 1:
                        # Save current metrics as standard
                        best_valid_pos_loss = valid_pos_loss
                        best_valid_auc = valid_auc_
                        best_valid_recon_loss = valid_recon_loss

                        best_epoch = epoch
                        torch.save(IAF_flow.state_dict(), flow_path)
                        torch.save(decoder.state_dict(), decoder_path)
                        torch.save(nu.state_dict(), nu_path)
                        improved_str = "*"
    #                     prior_z = sample_mixedGPD(8000,  mu=IAF_flow.mu0, logvar=IAF_flow.logvar0,\
    #                                               xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
    #                                               p_ = u_bound, lower_bound = -5.0, upper_bound = 50, device=device)
    #                     view_distribution(batch_z, prior_z, model_name, plot_path)

                if (epoch - best_epoch >=10) and (epoch - last_shrink >=10):
                    lambda_[1] = lambda_[1] * 5e-1
                    lambda_[2] = lambda_[2] * 5e-1
                    last_shrink = epoch


                print('====> Valid BCE loss: {:.4f}\t Pos Recon Loss: {:.4f} KL Loss: {:.4f} AUC: {:.4f} \tImproved: {}'.format(valid_recon_loss, valid_pos_loss, KL_cond, valid_auc_, improved_str))

                if epoch - best_epoch >=30:
                    print('Model stopped due to early stopping')
                    break
                
        # report results in testing        
        pred_label_risk, batch_z, Hz, auc_, auprc_ = testing_VIEVT(test, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, transform = True, norm_mean=norm_mean, norm_std=norm_std, continuous_variables=continuous_variables, device=device, saveResults=True)
        # bootstrapping
        _auc, _auprc = boostrappingCI(test['e'], pred_label_risk, "VIE", N=1000, nseed=124)
        np.save(result_path+'/'+'VIEVT_bootstrap_auc', _auc)
        np.save(result_path+'/VIEVT_bootstrap_auprc', _auprc)
