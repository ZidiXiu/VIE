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
from torch.distributions import normal

import pandas as pd

# calculate cdf for each dimension
def normal_cdf(value, loc, scale, noise = 1e-4):
    return (0.5 * (1 + torch.erf((value - loc) * (scale+noise).reciprocal() / (math.sqrt(2))))).to(loc.device)
def normal_icdf(value, loc, scale):
    return (loc + scale * torch.erfinv(2 * value - 1) * (math.sqrt(2))).to(loc.device)
# Reconstruction + KL divergence losses summed over all elements and batch
# MVN log-likelihood
def MVN_loglikeli(z, mu, logvar, noise = 1e-8):
    # note that Sigma is a diagonal matrix and we only have the diagonal information here
    p = logvar.size()[1]
    varmatrix = logvar.exp()+noise
    # for diagonal matrix:
    const = (torch.tensor(2*math.pi).log()).to(z.device)
    loglikeli = -0.5*(varmatrix.log() + (z-mu).pow(2)/varmatrix)
     
    loglikeli_ = loglikeli- (0.5*const).expand(loglikeli.size())
    return torch.sum(loglikeli_,axis=1)

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
    p = z.size()[1]
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
    xi_flag = (((xi_.clone()-0.0).abs())>= 0.0)*1
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

### Try new output function
def loglog_function(z):
    p = 1-(-z.exp()).exp()
    return p


### sample from mixed GPD distribution
# generate samples from mixed GPD

# ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/generalized_pareto.py
def sample_GPD( u, sigma_, xi_, rande=None, device='cpu',eps=1e-4, n=1,seed=None):
# Inversion samples via inverse CDF.
# need to input rande \sim UNIF(0,1)
#     z_dim = loc.size()[0]
    # indicator function regarding to xi_
    if type(rande)==type(None):
        torch.manual_seed(seed)
        rande = (torch.Tensor(n, sigma_.size()[0]).uniform_(0, 1)).to(xi_.device)
        
    xi_fake = (xi_.clone()).to(xi_.device)
    xi_flag = ((((xi_-0.0).abs())<=eps)).to(xi_.device)
    xi_fake[xi_flag] = eps
    xi_flag = (((xi_fake-0.0).abs())>eps)*1

    loge = (torch.log1p(-rande)).to(xi_.device)
    where_nonzero = u + (sigma_ / xi_fake) * torch.expm1(-xi_fake * loge)
    where_zero = u - sigma_ * loge
    
    rand_out = xi_flag*where_nonzero + (1-xi_flag)*where_zero
    
    return rand_out

def sample_normal(mu, logvar, rande = None, device='cpu',n=1,seed=None):
#         std = torch.exp(0.5*logvar)
    if type(rande)==type(None):
        torch.manual_seed(seed)
        rande = (torch.Tensor(n, sigma_.size()[0]).uniform_(0, 1)).to(mu.device)

    stddev = (logvar.exp().sqrt()).to(mu.device)

    N_standard = (normal.Normal(0.0, 1.0))
    Phi_e = (N_standard.icdf(rande)).to(mu.device)

    return (mu + Phi_e*stddev).to(mu.device)

def sample_mixedGPD(n,  mu, logvar, xi_, sigma_, p_ = 0.95, eps=1e-4, seed = 123,device='cpu', lower_bound = -5.0, upper_bound = 100):
    # n: number of samples to draw
    # mu, logvar: parameters of normal distribution
    # xi_, sigma_: parameters of GPD distribution
    # p_: tail part boundary, use to decide u (location parameter) in mixed GPD
    p = xi_.size()[0]
    if type(seed)==type(None):
        torch.manual_seed(seed)
    rande = (torch.Tensor(n, p).uniform_(0, 1)).to(device)
    u = torch.distributions.Normal(mu, torch.exp(0.5 * logvar)).icdf((torch.tensor(p_).expand(rande.size())).to(device)).to(device)
#     u = normal_icdf((torch.tensor(p_).expand(rande.size())).to(device), mu, (logvar.exp()).sqrt()).to(device)
    
    # if the random generated number greater than the threshold
    tail_flag = ((rande>p_)*1).to(mu.device)
    
    # sample from Normal distribution
    rand_normal = sample_normal(mu, logvar, rande, device)
    
    # calculate new sigma_ and u
    mixed_u, mixed_sigma =  mixed_GPD_params(u, p_, xi_, sigma_, eps=eps)
    
    # sample from GPD distribution with updated parameters
    # u, sigma_, xi_, rande=None, n=1,device='cpu',eps=1e-4, seed=None
    rand_GPD = sample_GPD(u=mixed_u, sigma_=mixed_sigma, xi_=xi_, rande=rande, device=mu.device, eps=eps)

    rand_out = tail_flag*rand_GPD.to(mu.device) + (1-tail_flag)*rand_normal.to(mu.device)
    
    # clamp extreme value
    
    rand_out = (torch.clamp(rand_out, min=lower_bound+1e-4, max=upper_bound)).to(mu.device)
    return rand_out

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)

    