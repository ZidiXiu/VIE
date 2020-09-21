import argparse
import numpy as np
import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

# imbalanced sampler

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset['e']))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return self.callback_get_label(dataset, idx)

                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
    
def callback_get_label(dataset, idx):
    # return label with the index provided
    # compatiable with EVT dataset format
    return dataset['e'][idx]

# # read from csv file
class EVTDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None, norm_mean = 0.0, norm_std = 1.0, continuous_variables=[]):
        """
        Args:
            csv_file (string): Path to the csv file of datasets.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        id_e = self.data_frame.iloc[idx, 0]
        id_x = self.data_frame.iloc[idx, 1:].as_matrix() 
        id_x = np.array(id_x)
        
        # sample = {'e': np.array([id_e]), 'x': id_x}

        if self.transform:
            # sample = self.transform(sample)
            id_x[continuous_variables] = (id_x.copy()[continuous_variables] - norm_mean) / norm_std

        return {"e":np.array([id_e]), "x": id_x}

# inherit torch.utils.data.Dataset for custom dataset
class EVTDataset_dic(Dataset):
    """normalizing continuous input"""

    def __init__(self, data_dictionary, transform=None, norm_mean = 0.0, norm_std = 1.0, continuous_variables=[]):
        """
        Args:
            csv_file (string): Path to the csv file of datasets.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dictionary = data_dictionary
        self.transform = transform
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.continuous_variables = continuous_variables

    def __len__(self):
        return len(self.data_dictionary['x'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        id_e = self.data_dictionary['e'][idx]
        id_x = self.data_dictionary['x'][idx,:]
        id_x = np.array(id_x)
        
        # sample = {'e': np.array([id_e]), 'x': id_x}

        if self.transform:
            # sample = self.transform(sample)
            id_x[self.continuous_variables] = (id_x.copy()[self.continuous_variables] - self.norm_mean) / self.norm_std

        return {"e":np.array([id_e]), "x": id_x}
    

