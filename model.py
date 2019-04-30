import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.backends import cudnn

from sklearn import manifold
import matplotlib.pyplot as plt

with open("data.pickle", 'rb') as f:
    x_train, x_test, y_train, y_test = pickle.load(f)

class TrainDataset(Dataset):
    
    def __init__(self):
        self.len = x_train.shape[0]
        self.x_data = torch.from_numpy(x_train.values)
        self.y_data = torch.from_numpy(y_train.values)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class TestDataset(Dataset):
    
    def __init__(self):
        self.len = x_test.shape[0]
        self.x_data = torch.from_numpy(x_test.values)
        self.y_data = torch.from_numpy(y_test.values)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

trainset = TrainDataset()
testset = TrainDataset()

## define our indices -- our dataset has 9 elements and we want a 8:4 split
num_train = len(trainset)
indices = list(range(num_train))
split = int(0.001*num_train)

# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

# Contiguous split
# train_idx, validation_idx = indices[split:], indices[:split]

## define our samplers -- we use a SubsetRandomSampler because it will return
## a random subset of the split defined by the given indices without replaf
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)

batch_size = 32

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # define the embedding
        self.id = nn.Embedding(100, 10)
        self.year = nn.Embedding(3, 2)
        self.month = nn.Embedding(12, 6)
        self.day = nn.Embedding(31, 10)
        self.hour = nn.Embedding(24, 10)
        self.dayofweek = nn.Embedding(7, 3)
        self.aqi = nn.Embedding(281, 12)
        self.humidity = nn.Embedding(81, 10)
        self.temp = nn.Embedding(46, 10)
        self.weather = nn.Embedding(13, 6)
        self.wind = nn.Embedding(9, 5)
        self.winp = nn.Embedding(10, 5)
        self.holiday = nn.Embedding(10, 6)
        self.surrounding = nn.Embedding(8, 3)
        
        # nn layers
        self.linear = nn.Linear(98, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 1)
        self.droplow = nn.Dropout3d(0.2)
        self.dropmid = nn.Dropout3d(0.5)
        self.drophigh = nn.Dropout3d(0.7)
        self.bnl1 = nn.BatchNorm1d(2048)
        self.bnl2 = nn.BatchNorm1d(1024)
        # nn.GRU()
        
    def forward(self, x):
        
        in_size = x.size(0)
        
        e_id = self.id(x[:,0])
        e_year = self.year(x[:,1])
        e_month = self.month(x[:,2])
        e_day = self.day(x[:,3])
        e_hour = self.hour(x[:,4])
        e_dayofweek = self.dayofweek(x[:,5])
        e_aqi = self.aqi(x[:,6])
        e_humidity = self.humidity(x[:,7])
        e_temp = self.temp(x[:,8])
        e_weather = self.weather(x[:,9])
        e_wind = self.wind(x[:,10])
        e_winp = self.winp(x[:,11])
        e_holiday = self.holiday(x[:,12])
        e_surrounding = self.surrounding(x[:,13])
        
        x = F.relu(torch.cat((e_id,e_year,e_month,e_day,e_hour,e_dayofweek,e_aqi,
                  e_humidity,e_temp,e_weather,e_wind,e_winp,e_holiday,e_surrounding), dim=1))
        
        x = F.relu(self.linear(x))
        x = self.dropmid(x)
        x = F.relu(self.linear2(x))
        x = self.dropmid(x)
        x = self.linear3(x)
        x = x.view(in_size)
        
        return x

model = Model()
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

LossRecord = []


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2))).data[0]
    
def mse(y, y_hat):
    """Compute root mean squared error"""
    return torch.mean((y - y_hat).pow(2)).data[0]

def train(epochs):
    model.train()
    for epoch in range(1, epochs + 1):

    	# Create the train_loader -- use your real batch_size which you
    	# I hope have defined somewhere above
    	train_loader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)

    	# You can use your above batch_size or just set it to 1 here.  Your validation
    	# operations shouldn't be computationally intensive or require batching.
    	validation_loader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=validation_sampler,num_workers=2)

    	# You can use your above batch_size or just set it to 1 here.  Your test set
    	# operations shouldn't be computationally intensive or require batching.  We 
    	# also turn off shuffling, although that shouldn't affect your test set operations
    	# either
    	# test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)

        # train the model
        loss_train = []
        for i, (x_data, y_data) in enumerate(train_loader, 0):
            x_data, y_data = Variable(x_data).cuda(), Variable(y_data.float()).cuda()
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            loss_train.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        loss_valid = []
        for i, (x_data, y_data) in enumerate(validation_loader,0):
            x_data, y_data = Variable(x_data).cuda(), Variable(y_data.float()).cuda()
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            loss_valid.append(loss.item())
        
        TrainLoss = torch.tensor(loss_train).mean().item()
        ValidLoss = torch.tensor(loss_valid).mean().item()
        LossRecord.append([TrainLoss, ValidLoss])

        print('Epoch:{} | Trian Loss: {:10.6f} Valid Loss: {:10.6f}'.format(epoch, TrainLoss, ValidLoss))


# train(10)
# Epoch:1 | Trian Loss: 103.786476 Valid Loss:  78.506714
# Epoch:2 | Trian Loss:  70.956413 Valid Loss:  63.247280
# Epoch:3 | Trian Loss:  60.296860 Valid Loss:  59.989174
# Epoch:4 | Trian Loss:  54.164146 Valid Loss:  54.034256
# Epoch:5 | Trian Loss:  50.699997 Valid Loss:  51.348503
# Epoch:6 | Trian Loss:  48.050747 Valid Loss:  49.381245
# Epoch:7 | Trian Loss:  46.131317 Valid Loss:  51.115108
# Epoch:8 | Trian Loss:  44.519798 Valid Loss:  47.709774
# Epoch:9 | Trian Loss:  43.398769 Valid Loss:  47.027718
# Epoch:10 | Trian Loss:  42.474014 Valid Loss:  44.889729