import math
import os
import numpy as np

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

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


    
# hidden dim 2
class MyMonotoneNN_dim2(nn.Module):
    def __init__(self, in_d, hidden_layers, bias=True):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        
        # define a linear layer
        self.linearity = torch.nn.Linear(2, 1)
        
        
    def forward(self, z, N, lower_bound, output_Hz = False):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        Hz0 = torch.zeros(n_batch,1).to(z.device)
        Hz1 = torch.zeros(n_batch,1).to(z.device)
        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
        Hz = torch.cat([Hz0, Hz1], dim = 1)
        output = self.linearity(Hz)
        if output_Hz:
            return output, Hz
        else:
            return output

    
class MyMonotoneNN_strict_dim2(nn.Module):
    def __init__(self, in_d, hidden_layers, bias=True):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        
#         # define a linear layer
#         bias = torch.rand(1)
        self.bias = torch.nn.Parameter(torch.rand(1))
        
        
    def forward(self, z, N, lower_bound):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        Hz0 = torch.zeros(n_batch,1).to(z.device)
        Hz1 = torch.zeros(n_batch,1).to(z.device)
        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
        Hz = torch.cat([Hz0, Hz1], dim = 1)
        output = Hz.sum(1)+self.bias
        return output
    
    
# hidden dim 4
class MyMonotoneNN(nn.Module):
    def __init__(self, in_d, hidden_layers, bias=True):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        self.z2 = MLP_block(1, hidden_layers)
        self.z3 = MLP_block(1, hidden_layers)
        
        # define a linear layer
        self.linearity = torch.nn.Linear(4, 1)
        
        
    def forward(self, z, N, lower_bound, output_Hz = False):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        Hz0 = torch.zeros(n_batch,1).to(z.device)
        Hz1 = torch.zeros(n_batch,1).to(z.device)
        Hz2 = torch.zeros(n_batch,1).to(z.device)
        Hz3 = torch.zeros(n_batch,1).to(z.device)
        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
            Hz2.add_((self.z2(z_random[:,2, bin_j].view(n_batch,1))+1)*d_size[:,2].view(n_batch,1))
            Hz3.add_((self.z3(z_random[:,3, bin_j].view(n_batch,1))+1)*d_size[:,3].view(n_batch,1))
        Hz = torch.cat([Hz0, Hz1, Hz2, Hz3], dim = 1)
        output = self.linearity(Hz)
        if output_Hz:
            return output, Hz
        else:
            return output


# hidden dim 4
class MyMonotoneNN_strict(nn.Module):
    def __init__(self, in_d, hidden_layers, bias=True):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        self.z2 = MLP_block(1, hidden_layers)
        self.z3 = MLP_block(1, hidden_layers)
        
        # define a bias
        self.bias = torch.nn.Parameter(torch.rand(1))        
        
    def forward(self, z, N, lower_bound):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        Hz0 = torch.zeros(n_batch,1).to(z.device)
        Hz1 = torch.zeros(n_batch,1).to(z.device)
        Hz2 = torch.zeros(n_batch,1).to(z.device)
        Hz3 = torch.zeros(n_batch,1).to(z.device)
        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
            Hz2.add_((self.z2(z_random[:,2, bin_j].view(n_batch,1))+1)*d_size[:,2].view(n_batch,1))
            Hz3.add_((self.z3(z_random[:,3, bin_j].view(n_batch,1))+1)*d_size[:,3].view(n_batch,1))
        Hz = torch.cat([Hz0, Hz1, Hz2, Hz3], dim = 1)
        output = Hz.sum(1) + self.bias
        return output

    
# hidden dim 8
class MyMonotoneNN_dim8(nn.Module):
    def __init__(self, in_d, hidden_layers, bias=True):
        super().__init__()

        self.z0 = MLP_block(1, hidden_layers)
        self.z1 = MLP_block(1, hidden_layers)
        self.z2 = MLP_block(1, hidden_layers)
        self.z3 = MLP_block(1, hidden_layers)
        self.z4 = MLP_block(1, hidden_layers)
        self.z5 = MLP_block(1, hidden_layers)
        self.z6 = MLP_block(1, hidden_layers)
        self.z7 = MLP_block(1, hidden_layers)
        # define a linear layer
        self.linearity = torch.nn.Linear(8, 1)
        
        
    def forward(self, z, N, lower_bound, output_Hz = False):
        # latent variable z with dimentions dim_z
        # integration split number N
        n_batch, n_dim = z.size()
        z_clamp = torch.clamp(z, min=lower_bound)
        z_random, d_size = compute_trapz_random(z_clamp, N, lower_bound)

        
        Hz0 = torch.zeros(n_batch,1).to(z.device)
        Hz1 = torch.zeros(n_batch,1).to(z.device)
        Hz2 = torch.zeros(n_batch,1).to(z.device)
        Hz3 = torch.zeros(n_batch,1).to(z.device)
        Hz4 = torch.zeros(n_batch,1).to(z.device)
        Hz5 = torch.zeros(n_batch,1).to(z.device)
        Hz6 = torch.zeros(n_batch,1).to(z.device)
        Hz7 = torch.zeros(n_batch,1).to(z.device)

        for bin_j in np.arange(N):
            # since the activation of the last layer is elu, need to +1 in order to guarantee positiveness
            Hz0.add_((self.z0(z_random[:,0, bin_j].view(n_batch,1))+1)*d_size[:,0].view(n_batch,1))
            Hz1.add_((self.z1(z_random[:,1, bin_j].view(n_batch,1))+1)*d_size[:,1].view(n_batch,1))
            Hz2.add_((self.z2(z_random[:,2, bin_j].view(n_batch,1))+1)*d_size[:,2].view(n_batch,1))
            Hz3.add_((self.z3(z_random[:,3, bin_j].view(n_batch,1))+1)*d_size[:,3].view(n_batch,1))
            Hz4.add_((self.z4(z_random[:,4, bin_j].view(n_batch,1))+1)*d_size[:,4].view(n_batch,1)) 
            Hz5.add_((self.z5(z_random[:,5, bin_j].view(n_batch,1))+1)*d_size[:,5].view(n_batch,1)) 
            Hz6.add_((self.z6(z_random[:,6, bin_j].view(n_batch,1))+1)*d_size[:,6].view(n_batch,1)) 
            Hz7.add_((self.z7(z_random[:,7, bin_j].view(n_batch,1))+1)*d_size[:,7].view(n_batch,1)) 

        Hz = torch.cat([Hz0, Hz1, Hz2, Hz3, Hz4, Hz5, Hz6, Hz7], dim = 1)
        output = self.linearity(Hz)
        if output_Hz:
            return output, Hz
        else:
            return output
