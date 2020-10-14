import math
import os
import numpy as np

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from torch.distributions import normal
import sklearn.metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from networks.MonotonicNN_outHz import MyMonotoneNN, MyMonotoneNN_dim8, MyMonotoneNN_dim2
from utils.distributions import mixed_loglikeli, loglog_function, sample_mixedGPD
from utils.metrics import binary_cross_entropy

from networks.autoregressive import MADE

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_layer_MNN, hidden_layers=[], loglogLink=False):
        super(Decoder, self).__init__()
        # input z
        # output p(z|x)
        if z_dim == 2:
            self.mnn = MyMonotoneNN_dim2(1, hidden_layer_MNN)
        if z_dim == 4: 
            self.mnn = MyMonotoneNN(1, hidden_layer_MNN)
        if z_dim == 8: 
            self.mnn = MyMonotoneNN_dim8(1, hidden_layer_MNN)            
        
        net = []
        hs = [1] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  
        net = nn.Sequential(*net)
        self.net = net
        if loglogLink:
            self.out = loglog_function
        else:
            self.out = nn.Sigmoid()       
        
        
    def forward(self, z, N, lower_bound, output_Hz = False):
        if output_Hz:
            mnn_result, Hz = self.mnn(z, N, lower_bound, output_Hz)
            return self.out(mnn_result), Hz
        else:
            mnn_result = self.mnn(z, N, lower_bound)

            return self.out(mnn_result)

    
class Decoder_VAE(nn.Module):
    def __init__(self, z_dim, hidden_layers=[], loglogLink=False):
        super(Decoder_VAE, self).__init__()
        # input z
        # output p(z|x)                
        
        net = []
        hs = [z_dim] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  
        net = nn.Sequential(*net)
        self.net = net
        if loglogLink:
            self.out = loglog_function
        else:
            self.out = nn.Sigmoid()       
        
        
    def forward(self,z):
        result = self.net(z)
        
        return self.out(result)

class Encoder(nn.Module):
    def __init__(self, in_d, hidden_layers=[32,32], z_dim=4):
        super(Encoder, self).__init__()
        self.in_d = in_d
        self.z_dim = z_dim
        
        # input x, eps
        # instead of output z, now output mu, logvar and h
        net = []
        hs = [in_d] + hidden_layers + [z_dim*3]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  
        net = nn.Sequential(*net)
        self.net = net
        
    def reparametrize(self, mu, logvar):
        n_samples, z_dim = mu.size()
        eps_ =  (torch.Tensor(n_samples, z_dim).normal_()).to(mu.device)
        stddev = logvar.exp().sqrt()
        z = mu + eps_*stddev
        l = torch.distributions.Normal(mu, torch.exp(0.5 * logvar)).log_prob(z)
        #l = (stddev.log()+0.5*eps_.pow(2) + 0.5*(torch.tensor(2*math.pi).log()).to(z.device)).sum()
        return z, l.sum()
        
    def forward(self,x,eps=None):
        if type(eps) != type(None):
            input_= torch.cat((x, eps), dim=1)
        else:
            input_ = x
        output_ = self.net(input_)
#         mu_, logvar_, h_ = torch.split(output_, self.z_dim)
        mu_, logvar_, h_ = torch.split(output_, self.z_dim, dim=1)
        
        z, l = self.reparametrize(mu_, logvar_)

        return z, h_, l
    
# ref: https://www.ritchievink.com/blog/2019/11/12/another-normalizing-flow-inverse-autoregressive-flows/
class AutoRegressiveNN(MADE):
    def __init__(self, in_features, hidden_features, context_features):
        super().__init__(in_features, hidden_features)
        self.context = nn.Linear(context_features, in_features)
        # remove MADE output layer
        del self.layers[len(self.layers) - 1]

    def forward(self, z, h):
        return self.layers(z) + self.context(h)
    
    
class IAF_step(nn.Module):
    def __init__(self, z_dim=4, h_dim=4, auto_regressive_hidden=32):
        super(IAF_step, self).__init__()
        self.z_dim = z_dim
        self.s_t = AutoRegressiveNN(
            in_features=z_dim,
            hidden_features=auto_regressive_hidden,
            context_features=h_dim,
        )
        self.m_t = AutoRegressiveNN(
            in_features=z_dim,
            hidden_features=auto_regressive_hidden,
            context_features=h_dim,
        )
#         self._kl_divergence_ = 0
    
    def determine_log_det_jac(self, g_t):
        return torch.log(g_t+1e-6).sum()
    
    def forward(self, inputs):
        if isinstance(inputs, tuple):
            z, h, _kl_divergence_ = inputs
        else:
            z, h, _kl_divergence_ = inputs, torch.zeros_like(z), 0
            
        # initially s_t should be large, i.e. 1 or 2
        s_t = self.s_t(z,h)+1.5
        g_t = torch.sigmoid(s_t)
        m_t = self.m_t(z,h)        
        
#         # -log |det Jac|
        self._kl_divergence_ =  _kl_divergence_ - self.determine_log_det_jac(g_t)

        # transformation
        new_z = g_t * z + (1 - g_t) * m_t
        # keep input and output the same
        return (new_z, h, self._kl_divergence_)

class IAF(nn.Module):
    def __init__(self, input_size, z_dim=4, h_dim=4, hidden_layers=[32,32], auto_regressive_hidden=32, nstep=5, device='cpu'):
        super(IAF, self).__init__()
        # p(z) parameters mu0, logvar0, xi_ and sigma_
        # fix the first part as standard normal
        
        self.mu0 = torch.zeros(z_dim).to(device)
#         self.mu0 = torch.nn.Parameter(mu0)
        self.logvar0 = torch.zeros(z_dim).to(device)
#         self.logvar0 = torch.nn.Parameter(logvar0)     
        
        self.xi_ =torch.rand(z_dim).to(device)
        # initialize with standard exponential distribution
#         self.xi_ = torch.zeros(z_dim)
        self.xi_ = torch.nn.Parameter(self.xi_)
        self.sigma_ = torch.ones(z_dim).to(device)
        self.sigma_ = torch.nn.Parameter(self.sigma_)
        
        # define encoder
        self.encoder = Encoder(in_d = input_size, hidden_layers=hidden_layers, z_dim=z_dim)
        
        # define flow
        self.flow = torch.nn.Sequential(*[IAF_step(z_dim, h_dim, auto_regressive_hidden) for _ in range(nstep)])
        
    def forward(self, x, eps = None):
        z_init, h_, l_qzx_init = self.encoder(x, eps)
        best_z, h_, l_qzx = self.flow((z_init, h_, l_qzx_init))
        
        return best_z, l_qzx
    
# CVB critic
class Nu(nn.Module):
    # take z as input also could take (x,z) as input
    # output log(nu(z))
    def __init__(self, z_dim=4, ncov=0, hidden_layers=[32,32],marginal=True):
        super(Nu, self).__init__()

        net = []
        if marginal:
            hs = [z_dim] + hidden_layers + [1]
        else:
            hs = [z_dim+ncov] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
#                 nn.ReLU(),
                nn.Softplus(),
            ])
        net.pop()
        
        net = nn.Sequential(*net)
        self.net = net
        
        self.out = nn.Tanh()
#         weights_init(self)

    def forward(self, z, x=None, nu_lambda=1.0, upper_cut = 30):
        if type(x) == type(None):
            out = self.out(self.net(z))
        else:
            inputs = torch.cat((x, z), dim=1)
#             out = self.out(self.net(inputs))
            out = self.net(inputs)
        out = torch.clamp(out,max=upper_cut)
        
        return nu_lambda*out
    
def log_score_marginal(nu, z, mu, logvar, xi_, sigma_, p_ = 0.95, eps=1e-3, nu_lambda=1.0,device='cpu',train_nu=False, opt_nu=None, clipping_value=1e-3):
    # considering marginal p(z) and q(z)
    # nu: critic network
    # n: number of samples to draw
    # mu, logvar: parameters of normal distribution
    # xi_, sigma_: parameters of GPD distribution
    # p_: tail part boundary, use to decide u (location parameter) in mixed GPD
    
    # nu_lambda: weights for nu outputs
    # here specify the prior of z is dim=4 MVN
    # if train_nu==True, opt_nu: optimizer for nu must be provided
    
    # z: z^{t}(\xi_i)
    
    # log(nu(z|x))
    z_nu = torch.mean(nu(z=z,nu_lambda=nu_lambda))
    
    # sample prior z
    n_samples = z.size()[0]
    prior_z = sample_mixedGPD(n_samples,  mu, logvar, xi_, sigma_, p_ = p_, eps=eps, device=device)
    
    # nu(prior_z)
    pz_nu = torch.mean(torch.exp(nu(z=prior_z.float(),nu_lambda=nu_lambda)))
    nanFlag = 0
    
    try:
        assert (pz_nu != pz_nu).any()== False  
    except AssertionError:
        nanFlag = 1
        print("pz_nu got nan")
#         print(prior_z, torch.exp(nu(z=prior_z,nu_lambda=nu_lambda)))
        
    if train_nu:
        opt_nu.zero_grad()
        # not updating Encoder and Decoder at all!
        loss_nu = -torch.mean( nu(z=z.detach(),nu_lambda=nu_lambda) ) + torch.mean( torch.exp( nu(z=prior_z.float().detach(),nu_lambda=nu_lambda) ) )
#         try:
#             assert (loss_nu != loss_nu).any()== False  
#         except AssertionError:
#             nanFlag = 1
            
        loss_nu.backward()
        ##gradient clippping to  avoid nan
        torch.nn.utils.clip_grad_norm_(nu.parameters(), clipping_value)

        opt_nu.step()

        return z_nu, pz_nu, loss_nu.item(),nanFlag
    else:
        return z_nu, pz_nu,nanFlag

    
def log_score(nu, x, z, mu, logvar, xi_, sigma_, p_ = 0.95, eps=1e-3, nu_lambda=1.0,device='cpu',train_nu=False, opt_nu=None, clipping_value = 1e-3):
    # considering p(z) and q(z|x)

    # n: number of samples to draw
    # mu, logvar: parameters of normal distribution
    # xi_, sigma_: parameters of GPD distribution
    # p_: tail part boundary, use to decide u (location parameter) in mixed GPD
    
    # nu_lambda: weights for nu outputs
    # here specify the prior of z is dim=4 MVN
    # if train_nu==True, opt_nu: optimizer for nu must be provided
    
    # z: z^{t}(\xi_i)
    
    # log(nu(z|x))
    z_nu = torch.mean(nu(x=x, z=z, nu_lambda=nu_lambda))
    
    # sample prior z
    n_samples = z.size()[0]
    prior_z = sample_mixedGPD(n_samples,  mu, logvar, xi_, sigma_, p_ = p_, eps=eps, device=device, seed = None)
    
    # nu(prior_z)
    pz_nu = torch.mean(torch.exp(nu(x=x, z=prior_z,nu_lambda=nu_lambda)))
    
    try:
        assert (pz_nu != pz_nu).any()== False  
    except AssertionError:
        nanFlag = 1
        print("pz_nu got nan")
    
    
    if train_nu:
        opt_nu.zero_grad()
        # not updating Encoder and Decoder at all!
        loss_nu = -torch.mean( nu(x=x, z=z.detach(),nu_lambda=nu_lambda) ) + torch.mean( torch.exp( nu(x=x, z=prior_z.detach(),nu_lambda=nu_lambda) ) )
#         try:
#             assert (loss_nu != loss_nu).any()== False  
#         except AssertionError:
#             nanFlag = 1
#             print("Nu training got nan")

        loss_nu.backward()
        ##gradient clippping to  avoid nan
        torch.nn.utils.clip_grad_norm_(nu.parameters(), clipping_value)

        opt_nu.step()

        return z_nu, pz_nu, loss_nu.item()
    else:
        return z_nu, pz_nu      

################
def pred_avg_risk_z(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=20, N=100, lower_bound=-5.0):
    eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)
    z_init, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())    
    for i in range(n_avg-1):
        
        eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)
        batch_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
        z_init = batch_z + z_init
        likelihood_qzx = likelihood_qzx + likelihood_qzx
        
#     pred_risk_batch = decoder(z_init/n_avg, N, lower_bound)
        
    return z_init/n_avg, likelihood_qzx/n_avg

def pred_avg_risk(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=20, N=100, lower_bound=-5.0):
    # expectation of epsilon and of x
    eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)
    z_init, likelihood_qzx = pred_avg_risk_z(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=n_avg, N=N, lower_bound=lower_bound)
    pred_risk_init = decoder(z_init, N, lower_bound)
    for i in range(n_avg-1):
        
        eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)
        batch_z, likelihood_qzx = pred_avg_risk_z(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=n_avg, N=N, lower_bound=lower_bound)
        pred_risk_batch = decoder(batch_z, N, lower_bound)
        pred_risk_init = pred_risk_batch + pred_risk_init
        likelihood_qzx = likelihood_qzx + likelihood_qzx
                
    return pred_risk_init/n_avg, likelihood_qzx/n_avg

def testing_VIEVT(dataset, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, L=100, N=100, u_bound=0.99, lower_bound=-5.0,transform = None, norm_mean=0.0, norm_std=1.0, continuous_variables=[],roc_curv = False, pr_curv = False, saveResults=True, device='cpu', BCE_Loss=False, limit_x=20):
    # also present the BCE loss and Positive case loss
    IAF_flow.load_state_dict(torch.load(flow_path))
    IAF_flow.eval()
    decoder.load_state_dict(torch.load(decoder_path))
    decoder.eval()
    nu.load_state_dict(torch.load(nu_path))
    nu.eval()        
    with torch.no_grad():
        input_size = dataset['x'].shape[1]
        if transform:
            batched_x = dataset['x'].copy()
            batched_x[:,continuous_variables] = (dataset['x'].copy()[:,continuous_variables] - norm_mean)/norm_std
            n_samples = batched_x.shape[0]
            batched_x =  torch.tensor(batched_x)
        else:
            n_samples = batched_x.shape[0]
            batched_x =  torch.tensor(dataset['x'])
        batched_x = batched_x.to(device).view(-1, input_size)
        
        batched_x = torch.clamp(batched_x, max = limit_x, min=-12)
        batched_e =  torch.tensor(dataset['e']).to(device)

        
        eps_ = (torch.Tensor( n_samples, eps_dim).normal_()).to(device)
        batch_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
        assert (batch_z != batch_z).any()== False
        _, Hz = decoder(batch_z, N, lower_bound, output_Hz = True)
        # add average over eps_ to increase accuracy
        pred_risk_batch, likelihood_qzx= pred_avg_risk(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=20)
        
        assert (pred_risk_batch != pred_risk_batch).any()== False
        
        recon_loss, pos_recon_ = binary_cross_entropy(pred_risk_batch.float(), \
                                 batched_e.detach().float(), sample_weight=None, pos_acc=True)

        # based on marginal q
        z_nu, pz_nu,nanFlag = log_score_marginal(nu=nu, z=batch_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                                p_ = u_bound, eps=1e-3, device=device, train_nu=False)

        data_loss = (recon_loss + z_nu - pz_nu).item()
        pred_risk_batch = np.vstack(pred_risk_batch.cpu()).reshape(-1)
#         pred_label_batch = get_predicted_label(pred_risk_batch)
#         # overall accuracy
#         overall_acc = np.mean(pred_label_batch==np.array(dataset['e']))
#         # percentage of predicted accuracy for patients with events only
#         event_idx = np.where(np.array(dataset['e'])==1)
#         pos_acc = np.mean(pred_label_batch[event_idx]==1)
        auc_ = sklearn.metrics.roc_auc_score(dataset['e'],pred_risk_batch)
        fpr_, tpr_, _ = sklearn.metrics.roc_curve(dataset['e'],pred_risk_batch)
        
        precision_, recall_, thresholds_ = precision_recall_curve(dataset['e'],pred_risk_batch)
#         f1_score_ = F1_score(precision_, recall_, beta=1.0)
        auprc_ = average_precision_score(dataset['e'],pred_risk_batch)

    print('====> Test set loss: {:.4f}, reconstruction loss: {:.4f}'.format(data_loss, recon_loss.item()))
#     print('====> Test overall Accuracy: {:.4f}, Positive Cases Accuracy: {:.4f}'.format(overall_acc, pos_acc))
    print('====> Test AUC score: {:.4f}'.format(auc_))
    print('====> Test AUPRC score: {:.4f}'.format(auprc_))
#     print('====> Test F1 score: {:.4f}'.format(f1_score_ ))

    
    if saveResults:
        np.save(result_path+'/test_risk_'+model_name, pred_risk_batch)
    
    if roc_curv:
        roc_curv_ = (fpr_, tpr_)
        pr_curv_ = (precision_, recall_)
        return auc_, roc_curv_, auprc_, pr_curv_
    if BCE_Loss:
        return pred_risk_batch,batch_z, Hz, auc_, auprc_, recon_loss.item(), pos_recon_.item()
    else:
        return pred_risk_batch,batch_z, Hz, auc_, auprc_
    
    
def testing_VIEVT_balanced(dataset, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, L=100, N=100, u_bound=0.99, lower_bound=-5.0,transform = None, norm_mean=0.0, norm_std=1.0, continuous_variables=[],roc_curv = False, pr_curv = False, saveResults=True, device='cpu'):
    IAF_flow.load_state_dict(torch.load(flow_path))
    IAF_flow.eval()
    decoder.load_state_dict(torch.load(decoder_path))
    decoder.eval()
    nu.load_state_dict(torch.load(nu_path))
    nu.eval()
    np.random.seed(123)
    auc_list = []
    auprc_list = []
    recon_list = []
    pos_recon_list = []    
    # calculating balanced auc, auprc, recon loss and positive loss
    for i in range(10):
        with torch.no_grad():
            n_samples, input_size = dataset['x'].shape

            e_idx = np.where(dataset['e']==0)[0]
            ne_idx = np.where(dataset['e']==1)[0]
            sub_samples = len(e_idx)
            ne_sub_idx = np.random.choice(ne_idx, sub_samples)
            sub_idx = np.concatenate([e_idx, ne_sub_idx])

            if transform:
                batched_x = dataset['x'][sub_idx,:].copy()
                batched_x[:,continuous_variables] = (dataset['x'][sub_idx,:][:,continuous_variables].copy() - norm_mean)/norm_std
                n_samples = batched_x.shape[0]
                batched_x =  torch.tensor(batched_x)
            else:
                n_samples = batched_x.shape[0]
                batched_x =  torch.tensor(dataset['x'][sub_idx,:])
            batched_x = batched_x.to(device).view(-1, input_size)
            batched_e =  torch.tensor(dataset['e'][sub_idx]).to(device)


            eps_ = (torch.Tensor( n_samples, eps_dim).normal_()).to(device)
            batch_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
            assert (batch_z != batch_z).any()== False
            _, Hz = decoder(batch_z, N, lower_bound, output_Hz = True)
            # add average over eps_ to increase accuracy
            pred_risk_batch, likelihood_qzx= pred_avg_risk(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=20)

            assert (pred_risk_batch != pred_risk_batch).any()== False

            recon_loss, pos_recon_ = binary_cross_entropy(pred_risk_batch.float(), \
                                     batched_e.detach().float(), sample_weight=None, pos_acc=True)

            # based on marginal q
            z_nu, pz_nu,nanFlag = log_score_marginal(nu=nu, z=batch_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                    xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
                                    p_ = u_bound, eps=1e-3, device=device, train_nu=False)

            data_loss = (recon_loss + z_nu - pz_nu).item()
            pred_risk_batch = np.vstack(pred_risk_batch.cpu()).reshape(-1)
    #         pred_label_batch = get_predicted_label(pred_risk_batch)
    #         # overall accuracy
    #         overall_acc = np.mean(pred_label_batch==np.array(dataset['e']))
    #         # percentage of predicted accuracy for patients with events only
    #         event_idx = np.where(np.array(dataset['e'])==1)
    #         pos_acc = np.mean(pred_label_batch[event_idx]==1)
            auc_ = sklearn.metrics.roc_auc_score(dataset['e'][sub_idx],pred_risk_batch)
            fpr_, tpr_, _ = sklearn.metrics.roc_curve(dataset['e'][sub_idx],pred_risk_batch)

            precision_, recall_, thresholds_ = precision_recall_curve(dataset['e'][sub_idx],pred_risk_batch)
    #         f1_score_ = F1_score(precision_, recall_, beta=1.0)
            auprc_ = average_precision_score(dataset['e'][sub_idx],pred_risk_batch)
            auc_list.append(auc_)
            auprc_list.append(auprc_)
            recon_list.append(recon_loss.item())
            pos_recon_list.append(pos_recon_.item())  
        print('====> Test set loss: {:.4f}, reconstruction loss: {:.4f}'.format(data_loss, recon_loss.item()))
    #     print('====> Test overall Accuracy: {:.4f}, Positive Cases Accuracy: {:.4f}'.format(overall_acc, pos_acc))
        print('====> Test AUC score: {:.4f}'.format(auc_))
        print('====> Test AUPRC score: {:.4f}'.format(auprc_))
    #     print('====> Test F1 score: {:.4f}'.format(f1_score_ ))


        if saveResults:
            np.save(result_path+'/test_risk_'+model_name, pred_risk_batch)
    
    if roc_curv:
        roc_curv_ = (fpr_, tpr_)
        pr_curv_ = (precision_, recall_)
        return auc_, roc_curv_, auprc_, pr_curv_

    else:
        return pred_risk_batch,batch_z, Hz, np.mean(np.array(auc_list)), np.mean(np.array(auprc_list)), np.mean(np.array(recon_list)), np.mean(np.array(pos_recon_list))