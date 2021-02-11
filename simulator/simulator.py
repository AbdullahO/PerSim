import numpy as np
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import random
from envs.base import Env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchvision import datasets, transforms
import pickle

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


####### To Dos #########

class Simulator_net(torch.nn.Module):
    def __init__(self, d_u, d_i,d_m, R, lags = 1,  continous_action = True, means = 0, stds = 1, means_state = None, stds_state = None, state_layers = [256,256], action_layers = [50], delta = True):
        super(Simulator_net, self).__init__()
        self.lin_u = nn.Linear(d_u, R)
        
        self.continous_action = continous_action
        
        self.action_layers = nn.ModuleList()
        current_dim = d_i
        
        if not self.continous_action:
            self.action_layers.append(nn.Linear(current_dim, R))
        
        else:            
            for hdim in action_layers:
                self.action_layers.append(nn.Linear(current_dim, hdim))
                current_dim = hdim
            self.action_layers.append(nn.Linear(current_dim, R))
            
        self.state_layers = nn.ModuleList()
        current_dim = d_m*lags
        for hdim in state_layers:
            self.state_layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.state_layers.append(nn.Linear(current_dim, d_m*R))
        if means_state is None:
            means_state = torch.zeros(d_m*lags)
            stds_state = torch.ones(d_m*lags)
        if means is None:
            means = torch.zeros(d_m)
            stds = torch.ones(d_m)
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)
        self.register_buffer('means_state', means_state)
        self.register_buffer('stds_state', stds_state)
        self.delta = delta 
        self.R = R
        self.d_m = d_m

    def forward(self, args):
        (u, i, m) = args
        u = self.lin_u(u)
        m = (m-self.means_state)/self.stds_state
        
        for layer in self.action_layers[:-1]:
            i = F.relu(layer(i))
        i = self.action_layers[-1](i)
        
        for layer in self.state_layers[:-1]:
            m = F.relu(layer(m))
        m = self.state_layers[-1](m)
        
        n = m.shape[0]
        m = m.view(n, self.d_m, self.R)
        m = m.permute(1, 0, 2)
        out = u*i*m
        out = out.sum(axis = 2)
        out = out.T
        
        return out
    
class Simulator(Env):
    def __init__(self, number_of_units, action_d, state_d, rank, device, continous_action = True, state_layers = [256], action_layers = [50], batch_size = 256, shuffle = True, lags = 5, means = None, stds = None, means_state = None, stds_state = None, delta = True):
        self.number_of_units = number_of_units
        self.action_d = action_d
        self.state_d = state_d
        self.rank = rank
        self.device = device 
        self.lags = lags
        self.model = Simulator_net(number_of_units, action_d, state_d, rank, lags = lags,  state_layers = state_layers, action_layers = action_layers,  continous_action = continous_action, means = means, stds = stds, means_state=means_state, stds_state = stds_state, delta = delta )
        self.model = self.model.to(self.device)
        self.state = torch.zeros(1,lags*state_d).to(self.device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.continous_action = continous_action
        self.delta = delta
       
    def step(self,state, action, unit = 0):
        ## 
        # state: tensor [ _ x state_dim*lags]
        
        n = state.shape[0]

        # get unit vector
        units = torch.zeros([n,self.number_of_units], device = self.device)
        units[:,unit] = 1
        
        if not self.continous_action:
            action_ = torch.zeros([n,self.action_d]).to(self.device)
            action_[torch.arange(n),action.long()[:,0]] =  1
        else:
            action_ = action.to(self.device)
        
        current_state = state
        new_state = self.model((units, action_, current_state))
        #denormalize
        new_state = new_state * self.model.stds + self.model.means
        # delta
        if self.delta:
            new_state = new_state + state[:, :self.state_d]
        
        return new_state, 0,0, 0


    # def reset(self, state):
    #     old_state = self.state.clone()
    #     self.state[0,self.state_d:] = old_state[0,:-self.state_d]
    #     self.state[0,:self.state_d] = torch.tensor(state)

    def train(self, data, learning_rate=1e-3, it=300):
        # Parameters
        params = {'batch_size': self.batch_size,
                  'shuffle': self.shuffle}

        learning_rate = learning_rate
        iterations = it
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        data_ = [[data[0][i],data[1][i],data[2][i],data[3][i]] for i in range(len(data[0]))]
        # data_loader 
        data_loader = torch.utils.data.DataLoader(data_, **params)


        loss_fn = torch.nn.MSELoss(reduction='mean')
        for t in tqdm(range(iterations)):
            # Training
            for batch in data_loader:
                # Forward pass
                u_data, i_data,m_data, y_data = batch
                y_data = (y_data -self.model.means)/self.model.stds
                y_pred = self.model((u_data, i_data, m_data))

                # Compute and print loss
                
                # Zero the gradients before running the backward pass
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_data)
            
                # Backward pass
                loss.backward()
                optimizer.step()
            if (t+1)% 50 == 0:
              for g in optimizer.param_groups:
                  g['lr'] = 0.5*g['lr']


                  
    def save(self,save_path):
        torch.save(self.model.state_dict(), f"{save_path}.pkl")
                            
            
    def load(self, load_path):
        # print(f'load path {load_path}')
        # with open(f'{load_path}.pkl', 'rb') as o:
        #     sim =  pickle.load(o)  
        # return sim 
        self.model.load_state_dict(torch.load(f'{load_path}.pkl', map_location=self.device))
            

