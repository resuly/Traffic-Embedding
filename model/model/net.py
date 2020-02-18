"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EM(nn.Module):

    def __init__(self, params):
        super(EM, self).__init__()

        self.params = params
        
        # define the embedding
        self.id = nn.Embedding(100, 150)
        self.year = nn.Embedding(3, 2)
        self.month = nn.Embedding(12, 15)
        self.day = nn.Embedding(31, 40)
        self.hour = nn.Embedding(24, 30)
        self.dayofweek = nn.Embedding(7, 15)
        self.aqi = nn.Embedding(281, 100)
        self.humidity = nn.Embedding(81, 100)
        self.temp = nn.Embedding(46, 100)
        self.weather = nn.Embedding(13, 15)
        self.wind = nn.Embedding(9, 15)
        self.winp = nn.Embedding(10, 15)
        self.holiday = nn.Embedding(10, 15)
        self.surrounding = nn.Embedding(8, 15)
        
        # nn layers
        self.linear = nn.Linear(627, 2048)
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
        # print(type(x[:,0]))
        # print(x[:,0])
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
        

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples

'''
def accuracy(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)
'''

def rmse(predictions, targets):
    """Compute root mean squared error"""
    return np.sqrt(((predictions - targets) ** 2).mean())

def mse(predictions, targets):
    """Compute mean squared error"""
    return ((predictions - targets) ** 2).mean()

def mae(predictions, targets):
    """Compute mean absolute error"""
    return np.sum(np.absolute((predictions - targets)))

def mape(predictions, targets):
    """Compute mean absolute precentage error"""
    mask = targets != 0
    return (np.fabs(targets[mask] - predictions[mask])/targets[mask]).mean()
    # return np.mean(np.absolute((targets - predictions) / targets)) * 100

def accuracy(predictions, targets):
    return 1 - mape(predictions, targets)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'mape': mape,
    'accuracy': accuracy
}
