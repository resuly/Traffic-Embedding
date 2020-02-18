import random
import os
import pandas as pd
import torch
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

with open("data/data_sample.pickle", 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

class PEMSTrianDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, params):
        self.len = x_train.shape[0]
        self.x_data = torch.from_numpy(x_train.values)
        self.y_data = torch.from_numpy(y_train.values)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

class PEMSTestDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, params):
        self.len = x_test.shape[0]
        self.x_data = torch.from_numpy(x_test.values)
        self.y_data = torch.from_numpy(y_test.values)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    dataloaders = {}
    
    # get the train_dataset
    train_dataset = PEMSTrianDataset(params=params)

    # print('train length:', len(train_dataset)) #train length: 4140

    # Define the indices
    indices = list(range(len(train_dataset))) # start with all the indices in training set
    split = int(0.1*len(train_dataset)) # define the split size
    # print('split', split)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader -- use your real batch_size which you
    # I hope have defined somewhere above
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, sampler=train_sampler)

    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, sampler=validation_sampler)

    dataloaders['train'] = train_loader
    dataloaders['val']   = validation_loader

    if 'test' in types:
        test_dataset = PEMSTestDataset(params=params)
        test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch_size)
        dataloaders['test'] = test_loader

    return dataloaders