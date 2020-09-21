# simulation a deep learning based dataset
from __future__ import absolute_import, division, print_function

import math
import os


import numpy as np
import pandas

import seaborn as sns

import matplotlib.pyplot as plt
import torch
import sklearn.metrics
import pandas as pd

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
from pathlib import Path

def split_t(times, p_bounds):
    bounds = np.percentile(times, p_bounds)
    nsplit = len(bounds)+1
    # the first group
    label0 = [1*(times<bounds[0])]
    # the last group
    labeln = [1*(times>bounds[-1])]
    label_list = []
    label_list.append(label0)
    for i in range(nsplit-2):
        idx = i+1
        label_ = [((1*(times>=bounds[idx-1]) + 1*(times<bounds[idx]))==2)*1]
        label_list.append(label_)
    label_list.append(labeln)
    
    label_list = np.array(label_list)
    label_list=np.transpose(label_list)
    outcome_list = np.arange(nsplit)
    final_label = np.matmul(label_list,outcome_list).squeeze()
    return final_label
        
def count_rates(test_e):
    ids=np.unique(test_e) #array of unique ids
    event_rate=np.array([len(test_e[test_e==i]) for i in ids])/len(test_e)
    return(event_rate)
    
        
def generate_data_semi(file_path, result_path, cut_bounds, seed, z_dim, tail=False, return_gx=False):
    
    x_pool = np.load(file_path+'framingham_x_pool'+'.npy')
    n_samples, ncov = x_pool.shape
    covariates = np.array(['AGE', 'HDL', 'TC', 'HRX', 'SYSBP', 'DIAB', 'SEX_0', 'SEX_1',
           'RACE_C_Black', 'RACE_C_Chinese American', 'RACE_C_Hispanic',
           'RACE_C_White', 'CURRSMK_0.0', 'CURRSMK_1.0'])
    
    categorical_variables = np.concatenate([np.array([3]),np.arange(5,14)])
    continuous_variables = np.array([0,1,2,4])
    encoded_indices = [[6, 7], [8, 9, 10, 11], [12, 13]]
    ref_cov = [cov[-1] for cov in encoded_indices]

    input_size = ncov
    np.random.seed(seed)
    perm_idx = np.random.permutation(n_samples)
    train_idx = perm_idx[0:int(n_samples/2)]
    valid_idx = perm_idx[int(n_samples/2):int(5*n_samples/6)]
    test_idx = perm_idx[int(5*n_samples/6):n_samples]
    
    if z_dim > 0:
        torch.manual_seed(139)
        model = Net3(input_size=input_size, h_dim=[4
], z_dim=z_dim, loglogLink=True, positive=True)

    #     optimizer = optim.Adam(model.parameters(), lr=5e-4)
        # data_name = 'fram_VIEVT_er05'

        Path(result_path).mkdir(parents=True, exist_ok=True)

        model_name = 'simulation_semi_multi'
        model_path = result_path+"/"+model_name+".pt"
        torch.save(model.state_dict(), model_path)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)        
                batched_x = x_pool.copy()
        norm_mean = np.mean(x_pool[:,continuous_variables],axis=0)
        norm_std = np.std(x_pool[:,continuous_variables],axis=0)

        batched_x[:,continuous_variables] = (x_pool.copy()[:,continuous_variables] - norm_mean)/norm_std
        xbeta=model(torch.Tensor(batched_x))
        xbeta = xbeta.detach().numpy()
#         gx = np.exp(xbeta[:,0])+ 3*np.exp(xbeta[:,1])
#         df = simulation_cox_weibull(gx, batched_x, lambda_=1e-2, nu_=6,cut_bound=cut_bound, seed=seed)
        gx = np.exp(xbeta[:,0]+1)+ 3*np.exp(4*xbeta[:,1])
        if tail:
            df = simulation_cox_weibull_tail(gx, x_pool, lambda_=1e-2, nu_=2,cut_bound=0, seed=seed)
        else: 
            df = simulation_cox_weibull(gx, x_pool, lambda_=1e-2, nu_=2,cut_bound=0, seed=seed)
        print(df)
        
        # split df['t']
        df['e'] = split_t(df['t'], cut_bounds)


    if return_gx:
        return gx, df['e']
    else:
        return df, continuous_variables
    
def generate_data(file_path, result_path, cut_bound, seed, n_samples = 50000, cox = True, return_gx=False):


    ncov = 5
    input_size = ncov
    torch.manual_seed(127)
    input_size = 5
    z_dim=2
    model = Net(output_size=input_size, h_dim=[32], z_dim=z_dim, loglogLink=True, positive=True)

    Path(result_path).mkdir(parents=True, exist_ok=True)

    model_name = 'simulation'
    model_path = result_path+"/"+model_name+".pt"
    print(model)
    torch.manual_seed(113)
    mu = torch.zeros(z_dim)
    logvar = torch.zeros(z_dim)
    xi_ = torch.rand(z_dim)
    sigma_ = torch.rand(z_dim)*2
    print(xi_, sigma_)
    # generate latent z
    z = sample_mixedGPD(n_samples,  mu, logvar, xi_, sigma_, p_ = 0.99, eps=1e-4, seed = 123,device='cpu', lower_bound = -5.0, upper_bound = 20)
    # generate x
    torch.save(model.state_dict(), model_path)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    torch.manual_seed(123)
    x, xbeta=model(z, N=100, lower_bound = -5.0)
    if cox:
        df = simulation_cox_weibull(xbeta.detach().numpy().squeeze(), x.detach().numpy().squeeze(),  lambda_=0.001, nu_=5,cut_bound=cut_bound, seed=seed)
    else:
        df = simulation_simple(xbeta.detach().numpy().squeeze(), x.detach().numpy().squeeze(),cut_bound=cut_bound, seed=seed)
    
    
    continuous_variables = np.arange(ncov)
    df['z'] = z
    if return_gx:
        return xbeta.detach().numpy().squeeze(), df['e']
    else:
        return df, continuous_variables

def simulation_cox_weibull(z, X, lambda_=0.5, nu_=1/100,cut_bound=1, seed=123):
    # linear relationship
    # n: number of patients        
    # the problem with the left tail is about this U!
    np.random.seed(seed)
    n = len(z)
    U = np.random.uniform(size=n)
#     U = 0.5
    # generate T accordingly
    T = (-np.log(U)/(lambda_*np.exp(z)))**(1/nu_)
    event = T<cut_bound
    
    return({"e":event*1, "x":X, "t": T})

def simulation_simple(z, X, cut_bound, seed=123):
    # linear relationship
    # n: number of patients        
    # the problem with the left tail is about this U!
    np.random.seed(seed)
    n = len(z)
    eps = np.random.normal(size=n)
#     U = 0.5
    # generate T accordingly
#     T = (-np.log(U)/(lambda_*np.exp(z)))**(1/nu_)
    T = -np.log(z.reshape(n)) + eps
    event = T<cut_bound
    
    return({"e":event*1, "x":X, "t": T})

def formatted_data_simu(x, t, e, z, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    latent_z = np.array(z[idx], dtype=float)
    covariates = np.array(x[idx], dtype=float)

    #print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring, 'z':latent_z}
    return survival_data 

def formatted_data_simu_noz(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx], dtype=float)

    #print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data 

def saveDataCSV(data_dic, name, file_path):
    df_x = pandas.DataFrame(data_dic['x'])
#     one_hot_indices = data_dic['one_hot_indices']
#     # removing 1 column for each dummy variable to avoid colinearity
#     if one_hot_indices:
#         rm_cov = one_hot_indices[:,-1]
#         df_x = df_x.drop(columns=rm_cov)
    df_e = pandas.DataFrame({'e':data_dic['e']})
    df_z = pandas.DataFrame(data_dic['z'])
    df = pandas.concat([df_e, df_x, df_z], axis=1, sort=False)
    df.to_csv(file_path+'/'+name+'.csv', index=False)
    
def loadDataCSV(name, file_path):
    df = pandas.read_csv(file_path+'/'+name+'.csv')
    n_total = df.shape[1]
    z_dim = 4
    df_x = df.iloc[:,range(1,n_total-z_dim)]
    df_e = df.iloc[:,0]
    df_z = df.iloc[:,range(n_total-z_dim, n_total)]
    return({'x':np.array(df_x), 'e':np.array(df_e), 'z':np.array(df_z)})

# Monotone network part
def compute_trapz_random(z, N, lower_bound = -5.0):
    # from latent variable z to get random slected values in each bin, in order to compute integration
    # z: latent variable, [nbatch, latent_dim]
    n_samples, n_latent_dim = z.size()[0], z.size()[1]
    # chop z
    z = torch.clamp(z, min=lower_bound)
    # calculate bin width for each z
    d_size = (((z-lower_bound)/torch.tensor(N-1).float())).to(z.device)
    d_order = (torch.arange(N).repeat(1,n_samples * n_latent_dim).view(n_samples, n_latent_dim, N)).to(z.device)
    z_split = d_size.view(n_samples, n_latent_dim,1) * d_order + lower_bound
    # random select a point in each bin
    try:
        d_random = np.stack([[np.append(np.random.uniform(0,d.item(), N-1),[0]) for d in sample] for sample in d_size.cpu()])
        z_random = z_split + torch.tensor(d_random).to(z.device)
    except OverflowError as  error:
        # Output expected OverflowErrors.
        z_random = z_split
    return z_random.float(), d_size.float()

    
def MLP_block(in_d, hidden_layers):
    # define h_j()
    # output would always be positive, so when define graphs need + 1 
        net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.LeakyReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer
        net.append(nn.ELU())
        net = nn.Sequential(*net)
        return net
    
class MyMonotoneNN(nn.Module):
    def __init__(self, in_d, hidden_layers, positive=False):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        self.z2 = MLP_block(1, hidden_layers)
        self.z3 = MLP_block(1, hidden_layers)
        self.positive = positive
        # define a linear layer
        if positive == True:
            self.linearity = torch.Tensor(4,1).uniform_(0, 1)
        else:   
            self.linearity = torch.nn.Linear(4, 1)
        
        
    def forward(self, z,  N=100, lower_bound=-0.5):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        Hz0 = torch.zeros(n_batch,1)
        Hz1 = torch.zeros(n_batch,1)
        Hz2 = torch.zeros(n_batch,1)
        Hz3 = torch.zeros(n_batch,1)
        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
            Hz2.add_((self.z2(z_random[:,2, bin_j].view(n_batch,1))+1)*d_size[:,2].view(n_batch,1))
            Hz3.add_((self.z3(z_random[:,3, bin_j].view(n_batch,1))+1)*d_size[:,3].view(n_batch,1))
        Hz = torch.cat([Hz0, Hz1, Hz2, Hz3], dim = 1)
        if self.positive==True:
            output = torch.mm(Hz, self.linearity)
        else:
            output = self.linearity(Hz)
        return output
    
class MyMonotoneNN_dim2(nn.Module):
    def __init__(self, in_d, hidden_layers, positive=False):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        self.positive = positive
        # define a linear layer
        if positive == True:
            self.linearity = torch.Tensor(2,1).uniform_(0, 1)
        else:   
            self.linearity = torch.nn.Linear(2, 1)
        
        
    def forward(self, z, N=100, lower_bound=-0.5):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        Hz0 = torch.zeros(n_batch,1)
        Hz1 = torch.zeros(n_batch,1)
        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
        Hz = torch.cat([Hz0, Hz1], dim = 1)
        if self.positive==True:
            output = torch.mm(Hz, self.linearity)
        else:
            output = self.linearity(Hz)
        return output    
# output z with extreme value distribution
# calculate cdf for each dimension
def normal_cdf(value, loc, scale, noise = 1e-4):
    return 0.5 * (1 + torch.erf((value - loc) * (scale+noise).reciprocal() / (math.sqrt(2))))
def normal_icdf(value, loc, scale):
    return loc + scale * torch.erfinv(2 * value - 1) * (math.sqrt(2))
# Reconstruction + KL divergence losses summed over all elements and batch
# MVN log-likelihood

# Generalized Pareto
def GPD_loglikeli(z, u, xi_, sigma_, eps=1e-3, noise = 1e-4):
    # indicator function regarding to xi_
    xi_fake = xi_.clone()
    xi_flag = (((xi_-0.0).abs())<=eps)
    xi_fake[xi_flag] = eps
    xi_flag = (((xi_fake-0.0).abs())>eps)*1
    
#     loglikeli = xi_flag*(-(sigma_+noise).log() - (1/(xi_+noise) +1)*(((1+xi_*(z-u).abs()/(sigma_+noise))+noise).log())) + (1-xi_flag)*(-(sigma_+noise).log()-((z-u).abs()/(sigma_+noise)))
    loglikeli = xi_flag*(-(sigma_+noise).log() - (1/(xi_+noise) +1)*(((1+xi_*(z-u)/(sigma_+noise)).abs()).log())) + (1-xi_flag)*(-(sigma_+noise).log()-((z-u)/(sigma_+noise)))
    
    return loglikeli

# new parameters for mixed tail
def mixed_GPD_params(u, Fu, xi_, sigma_, eps=1e-4):
    # indicator function regarding to xi_
    xi_fake = xi_.clone()
    xi_flag = (((xi_-0.0).abs())<=eps)
    xi_fake[xi_flag] = eps
    xi_flag = (((xi_fake-0.0).abs())>eps)*1
    
    new_sigma = xi_flag*(sigma_*((1-Fu)**(xi_fake))) + (1-xi_flag)*sigma_
    new_u = xi_flag*(u - new_sigma*(((1-Fu)**(-xi_fake)) - 1)/xi_fake) + (1-xi_flag)*(u+new_sigma*(np.log(1-Fu)))
    
    return new_u, new_sigma

def mixed_loglikeli(z, mu, logvar, xi_, sigma_, p_ = 0.95, eps=1e-3, noise = 1e-4):
    # get loglikelihood for log p(z)
    # first calculate the loglikelihood with MVN distribution
    p = z.size()[0]
    varmatrix = logvar.exp()+noise
    # for diagonal matrix:
    const = (torch.tensor(2*math.pi).log()).to(z.device)
    loglikeli_MVN_ = -0.5*(varmatrix.log() + (z-mu).pow(2)/varmatrix)
    loglikeli_MVN = loglikeli_MVN_ - (0.5*const).expand(loglikeli_MVN_.size())
    
    # calculate F(z)
#     z_cdf = normal_cdf(value = z, loc = mu, scale = logvar.exp().sqrt())
    # calculate u = i_F(percentile), Fu = p_
    u = normal_icdf((torch.tensor(p_).expand(z.size())).to(z.device), mu, (logvar.exp()+noise).sqrt())
    
    # then calculate likelihood with GPD distribution
    tail_flag = (z>u)*1
    
    # calculate new sigma_ and u
    mixed_u, mixed_sigma =  mixed_GPD_params(u, p_, xi_, sigma_, eps=eps)
    # based on new sigma_ and u, calculate the likelihood for the mixed tail log-likelihood
    # need to cut z in order to fall into the support
    xi_flag = (((xi_.clone()-0.0)).abs()>= 0.0)*1
    new_z = (torch.zeros(z.size())).to(z.device)
    for j in np.arange(p):
        new_z[:,j] = xi_flag[j]*z[:,j] + (1-xi_flag[j])*torch.clamp(z[:,j], max=(mixed_u[0,j]-mixed_sigma[j]/xi_[j]).item())
    
    loglikeli_mixedGPD = GPD_loglikeli(new_z, mixed_u, xi_, mixed_sigma , eps=eps, noise = noise)
    
    # final loglikelihood
    loglikeli_ = tail_flag*loglikeli_mixedGPD + (1-tail_flag)*loglikeli_MVN
    
    # return
#     loglikeli = torch.sum(loglikeli_, axis=1)
    if (1*torch.isnan(loglikeli_)).sum() > 0:
        print(loglikeli_, xi_, sigma_)
    
    return loglikeli_

# generate samples from mixed GPD

# ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/generalized_pareto.py
def sample_GPD( u, sigma_, xi_, rande=None, device='cpu',eps=1e-4, n=1,seed=None):
# Inversion samples via inverse CDF.
# need to input rande \sim UNIF(0,1)
#     z_dim = loc.size()[0]
    # indicator function regarding to xi_
    if type(rande)==type(None):
        torch.manual_seed(123)
        rande = (torch.Tensor(n, sigma_.size()[0]).uniform_(0, 1)).to(device)
        
    xi_fake = xi_.clone()
    xi_flag = (((xi_-0.0).abs())<=eps)
    xi_fake[xi_flag] = eps
    xi_flag = (((xi_fake-0.0).abs())>eps)*1

    loge = (torch.log1p(-rande)).to(device)
    where_nonzero = u + (sigma_ / xi_fake) * torch.expm1(-xi_fake * loge)
    where_zero = u - sigma_ * loge
    
    rand_out = xi_flag*where_nonzero + (1-xi_flag)*where_zero
    
    return rand_out

def sample_normal(mu, logvar, rande = None, device='cpu',n=1):
#         std = torch.exp(0.5*logvar)
    if type(rande)==type(None):
        torch.manual_seed(123)
        rande = (torch.Tensor(n, sigma_.size()[0]).uniform_(0, 1)).to(device)

    stddev = (logvar.exp().sqrt()).to(device)

    N_standard = (normal.Normal(0.0, 1.0))
    Phi_e = (N_standard.icdf(rande)).to(device)

    return (mu + Phi_e*stddev).to(device)

def sample_mixedGPD(n,  mu, logvar, xi_, sigma_, p_ = 0.95, eps=1e-4, seed = 123,device='cpu', lower_bound = -5.0, upper_bound = 500):
    # n: number of samples to draw
    # mu, logvar: parameters of normal distribution
    # xi_, sigma_: parameters of GPD distribution
    # p_: tail part boundary, use to decide u (location parameter) in mixed GPD
    p = xi_.size()[0]
    if type(seed)!=type(None):
        torch.manual_seed(seed)
        rande = (torch.Tensor(n, p).uniform_(0, 1)).to(device)
    else:
        rande = (torch.Tensor(n, p).uniform_(0, 1)).to(device)
    
    u = normal_icdf((torch.tensor(p_).expand(rande.size())).to(device), mu, (logvar.exp()).sqrt())
    
    # if the random generated number greater than the threshold
    tail_flag = (rande>p_)*1
    
    # sample from Normal distribution
    rand_normal = sample_normal(mu, logvar, rande, device)
    
    # calculate new sigma_ and u
    mixed_u, mixed_sigma =  mixed_GPD_params(u, p_, xi_, sigma_, eps=eps)
    
    # sample from GPD distribution with updated parameters
    # u, sigma_, xi_, rande=None, n=1,device='cpu',eps=1e-4, seed=None
    rand_GPD = sample_GPD(u=mixed_u, sigma_=mixed_sigma, xi_=xi_, rande=rande, device=device, eps=eps)

    rand_out = tail_flag*rand_GPD + (1-tail_flag)*rand_normal
    
    # clamp extreme value
    
    rand_out = torch.clamp(rand_out, min=lower_bound+1e-4, max=upper_bound)
    return rand_out


# define the generating network

class Net(nn.Module):
    # input z which follows an extreme distribution
    # output x, xbeta
    def __init__(self, output_size=5, h_dim=[], z_dim=4, loglogLink=False, positive=False):
        super(Net, self).__init__()
        self.z_dim = z_dim
        self.ncov = output_size
        net = []
        
        hs = [z_dim] + h_dim + [output_size]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer
        self.mnn = MyMonotoneNN_dim2(1, [], positive)
#         net.append(nn.Sigmoid())

        self.net = nn.Sequential(*net)
        

    def forward(self, z, N, lower_bound):
        # generate x from latent z
        x = self.net(z)
        # generate xbeta to plug in cox model
        xbeta = self.mnn(z, N, lower_bound)
        return x, xbeta
    
class Net2(nn.Module):
    # output z directly
    def __init__(self, input_size=5, h_dim=[], z_dim=4, loglogLink=False, positive=False):
        super(Net2, self).__init__()
        self.z_dim = z_dim
        net = []
        
        hs = [input_size] + h_dim + [z_dim]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer
        if z_dim == 1:
            self.mnn = nn.Linear(1,1)
        if z_dim ==4:
            self.mnn = MyMonotoneNN(1, [], positive)
        if z_dim ==2:
            self.mnn = MyMonotoneNN_dim2(1, [], positive)
#         net.append(nn.Sigmoid())

        self.net = nn.Sequential(*net)
        

    def forward(self, x, N, lower_bound, seed = None):
        n, ncov = x.size()
        z_ = self.net(x)
        xbeta = self.mnn(z_)
        # clamp xbeta output range
        xbeta = torch.clamp(xbeta, max=30)
        
        return xbeta, z_

class Net3(nn.Module):
    # output z directly without MNN
    def __init__(self, input_size=5, h_dim=[], z_dim=2,loglogLink=False, positive=False):
        super(Net3, self).__init__()
        net = []
        
        hs = [input_size] + h_dim + [z_dim]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer

        self.net = nn.Sequential(*net)
        

    def forward(self, x, seed = None):
        n, ncov = x.size()
        z_ = self.net(x)
        # clamp xbeta output range
        xbeta = torch.clamp(z_, max=30)
        return xbeta
    

def nonlinear_xbeta(x_pool, continuous_variables=np.array([0,1,2,4])):
    x_pool_new = x_pool.copy()
    norm_mean = np.mean(x_pool[:,continuous_variables],axis=0)
    norm_std = np.std(x_pool[:,continuous_variables],axis=0)

    x_pool_new[:,2] = np.log(x_pool[:,2])
    x_pool_new[:,4] = ((x_pool[:,4]-norm_mean[3])/norm_std[3])**2
    x_pool_new[:,0] = np.exp((x_pool[:,0]-norm_mean[0])/norm_std[0])
    x_pool_new[:,1] = ((x_pool[:,1]-norm_mean[1])/norm_std[1])**2
    
    return x_pool_new 

# generate event from tail part
def simulation_cox_weibull_tail(z, X, lambda_=0.5, nu_=1/100,cut_bound=1, seed=123):
    # linear relationship
    # n: number of patients        
    # the problem with the left tail is about this U!
    np.random.seed(seed)
    n = len(z)
    U = np.random.uniform(size=n)
#     U = 0.5
    # generate T accordingly
    T = (-np.log(U)/(lambda_*np.exp(z)))**(1/nu_)
    event = T>cut_bound
    
    return({"e":event*1, "x":X, "t": T})
