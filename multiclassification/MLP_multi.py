from __future__ import absolute_import, division, print_function

import math
import os


import numpy as np
import pandas

import seaborn as sns

import matplotlib.pyplot as plt
import torch
import sklearn.metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
print ("PyTorch version: " + torch.__version__)

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

from utils.distributions import loglog_function


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleMLP(nn.Module):
    def __init__(self, input_size=5, h_dim=[32,32], z_dim=32, n_out=5,loglogLink=False):
        super(SimpleMLP, self).__init__()
        net = []
        hs = [input_size] + h_dim + [n_out]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer
#         if loglogLink:
#             self.out = loglog_function
#         else:
#             self.out = nn.Sigmoid()
#         net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
        

    def forward(self, x):
        output = self.net(x)
        return output


class MNN_MLP(nn.Module):
    def __init__(self, input_size=5, h_dim=[8,8], z_dim=8, hz_dim = [4], loglogLink=False):
        super(MNN_MLP, self).__init__()
        net = []
        hs = [input_size] + h_dim + [z_dim]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()  # pop the last ReLU for the output layer
#         net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
#         self.mnn = MyMonotoneNN(z_dim, hz_dim)
        if z_dim == 4:
            self.mnn = MyMonotoneNN(1, hz_dim)
            
        if z_dim == 8:
            self.mnn = MyMonotoneNN_dim8(1, hz_dim)
        
        if loglogLink:
            self.out = loglog_function
        else:
            self.out = nn.Sigmoid()
#         out.append(nn.Sigmoid())
#         self.out = nn.Sequential(*out)
        

    def forward(self, x, N=100, lower_bound=-5.0):
        z = (self.net(x)).to(x.device)
        y = self.mnn(z,N, lower_bound)
        
        return self.out(y)
    

from sklearn.metrics import confusion_matrix
def accuracy_per_class(predict, label):
    cm = confusion_matrix(label, predict)
    return np.diag(cm)/np.sum(cm, axis = 1)

def get_predicted_label(recon_y):
    # get the largest probability in the two categories
    pred_label = np.argmax(recon_y,axis=1)
    # return the label only
    return np.array(pred_label)

def loss_function(pred, target, sample_weight=None, class_acc = False):
    nc = pred.size()[1]
#     pred = pred.view(pred.shape[0], -1)
    
    # according to nn.crossentropyloss manual, sample_weight need to have shape [c], instead of [n]
    if type(sample_weight)==type(None):
        sample_weight = torch.ones(nc).to(pred.device)
    loss = nn.CrossEntropyLoss(weight = sample_weight)
    # pred shape = [N,C]
    output = loss(pred, target)
    if class_acc == True:
        pred_label =  get_predicted_label(pred.detach().cpu().numpy())
        class_acc_output = accuracy_per_class(pred_label.squeeze(), target.detach().cpu().numpy())
        return output, class_acc_output
    else: 
        return output
    
def testing_MLP(dataset, model, model_path, model_name, result_path, transform = None, norm_mean=0.0, norm_std=1.0, continuous_variables=[],roc_curv = False, pr_curv = False, saveResults=True, device='cpu'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_path))
    data_loss = 0
    model.eval()
    pred_label_all = []
    pred_risk_all = []
    input_size = dataset['x'].shape[1]
    with torch.no_grad():
        if transform:
            batched_x = dataset['x'].copy()
            batched_x[:,continuous_variables] = (dataset['x'].copy()[:,continuous_variables] - norm_mean)/norm_std
            batched_x =  torch.tensor(batched_x)
        else:
            batched_x =  torch.tensor(dataset['x'])
        batched_x = batched_x.to(device).view(-1, input_size)
        batched_e =  torch.tensor(dataset['e']).to(device)

        pred_risk_batch = model(x=batched_x.float())
        data_loss, test_class_acc= loss_function(pred_risk_batch.float(), batched_e.squeeze().long(),sample_weight=None, class_acc=True)
        pred_label_batch = get_predicted_label(pred_risk_batch.detach().cpu())
        test_F1 = sklearn.metrics.f1_score(pred_label_batch,\
                            batched_e.cpu().squeeze().detach().long(),average='micro')

    print('====> Test set loss: {:.4f}'.format(data_loss))
    print('====> Test Class Accuracy: {}, F1 score: {:.4f}'.format(test_class_acc, test_F1))

    
    if saveResults:
        np.save(result_path+'/test_risk_'+model_name, pred_risk_batch.cpu())
    
    if roc_curv:
        roc_curv_ = (fpr_, tpr_)
        pr_curv_ = (precision_, recall_)
        return auc_, roc_curv_, auprc_, pr_curv_

    else:
        return pred_risk_batch, test_class_acc, test_F1


def testing_MLP_balanced(dataset, model, model_path, model_name, result_path, transform = None, norm_mean=0.0, norm_std=1.0, continuous_variables=[],roc_curv = False, pr_curv = False, saveResults=True, device='cpu'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(model_path))
    data_loss = 0
    model.eval()
    pred_label_all = []
    pred_risk_all = []
    auc_list = []
    auprc_list = []
    recon_list = []
    pos_recon_list = []
    np.random.seed(123)

    input_size = dataset['x'].shape[1]
    
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
                batched_x =  torch.tensor(batched_x)
            else:
                batched_x =  torch.tensor(dataset['x'][sub_idx,:])
            batched_x = batched_x.to(device).view(-1, input_size)
            batched_e =  torch.tensor(dataset['e'][sub_idx]).to(device)

            pred_risk_batch = model(x=batched_x.float())
            data_loss, pos_loss= loss_function(pred_risk_batch.float(), batched_e.squeeze().long(),sample_weight=None, pos_acc=True)

        print('====> Test set loss: {:.4f}, poss loss: {:.4f}'.format(data_loss, pos_loss))
    #     print('====> Test overall Accuracy: {:.4f}, Positive Cases Accuracy: {:.4f}'.format(overall_acc, pos_acc))
        print('====> Test AUC score: {:.4f}'.format(auc_))
        print('====> Test AUPRC score: {:.4f}'.format(auprc_))
        auc_list.append(auc_)
        auprc_list.append(auprc_) 
        recon_list.append(data_loss.item())
        pos_recon_list.append(pos_loss.item())
    
    if saveResults:
        np.save(result_path+'/test_risk_'+model_name, pred_risk_batch)
    
    if roc_curv:
        roc_curv_ = (fpr_, tpr_)
        pr_curv_ = (precision_, recall_)
        return auc_, roc_curv_, auprc_, pr_curv_

    else:
        return pred_risk_batch, np.mean(np.array(auc_list)), np.mean(np.array(auprc_list)), np.mean(np.array(recon_list)), np.mean(np.array(pos_recon_list))

