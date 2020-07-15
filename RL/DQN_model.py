import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import torchvision.transforms
import matplotlib.pyplot as plt
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        # input format is (B,H,W,C)
        #with torch.no_grad():
        #    x = x.permute(0,3,1,2)
        x = x/255 # normalize
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, S):
        with torch.no_grad():
            S = Variable(torch.FloatTensor(np.float32(S))).unsqueeze(0)
            q_value = self.forward(S)
            a = q_value.max(1)[1].data[0]
        return a
    
    
    
class CnnDDQN(nn.Module):
    def __init__(self, input_shape, num_actions, prov_bias = None):
        super(CnnDDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions,  bias=False) # The bias will be from a shared layer between models
        )
        if prov_bias == None:
            self.bias = nn.Parameter(torch.zeros(num_actions)) # This is the shared bias layer
        else: 
            self.bias = prov_bias
            
    def return_bias(self):
        return self.bias
    
    def forward(self, x, bias = None):
        # input format is (B,H,W,C)
        #    x = x.permute(0,3,1,2)
        x = x/255 # normalize
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x + self.bias
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, S):
        with torch.no_grad():
            S = Variable(torch.FloatTensor(np.float32(S))).unsqueeze(0)
            q_value = self.forward(S)
            a = q_value.max(1)[1].data[0]
        return a
    

class CnnDQN_VAE(nn.Module):
    def __init__(self, input_shape, num_actions, max_pool=False):
        super(CnnDQN_VAE, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        if max_pool: # Use maxpool in case input size to large
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
                nn.ReLU()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, S):
        S   = Variable(torch.FloatTensor(np.float32(S)))#.unsqueeze(0))
        q_value = self.forward(S)
        a  = q_value.max(1)[1].data[0]
        return a
    
    
class CnnDDQN_VAE(nn.Module):
    def __init__(self, input_shape, num_actions, max_pool=False, prov_bias = None):
        super(CnnDDQN_VAE, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        if max_pool: # Use maxpool in case input size to large
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=0),
                nn.ReLU()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions,  bias=False)
        )
        if prov_bias == None:
            self.bias = nn.Parameter(torch.zeros(num_actions)) # This is the shared bias layer
        else: 
            self.bias = prov_bias
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def return_bias(self):
        return self.bias
    
    def act(self, S):
        # S   = Variable(torch.FloatTensor(S))#.unsqueeze(0))
        q_value = self.forward(S)
        a  = q_value.max(1)[1].data[0]
        return a
    
    def forward(self, x, bias = None):
        # input format is (B,H,W,C)
        # We assume data is normalized
        #    x = x.permute(0,3,1,2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x + self.bias
        return x
