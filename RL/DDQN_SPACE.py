#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:48:30 2020

@author: tirsgaard
# This is still outdated compared to the DDQN.py
"""

import math, random

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import torchvision.transforms
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

from IPython.display import clear_output
import matplotlib.pyplot as plt
from helper_functions import Replay_buffer, phi_transformer
from DQN_model import CnnDQN_VAE
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.benchmark=True
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

writer = SummaryWriter()



## Back propergation
def compute_td_loss(batch_size, iter_number):
    S, a, r, S_next, done = replay_buffer.return_batch(batch_size)

    S      = Variable(torch.FloatTensor(np.float32(S)))
    S_next = Variable(torch.FloatTensor(np.float32(S_next)))
    a      = Variable(torch.LongTensor(a))
    r      = torch.sign(Variable(torch.FloatTensor(r)))
    done   = Variable(torch.FloatTensor(done))

    
    with torch.no_grad():
        q_next_train = Q_train(S_next)
        q_next_target = Q_target(S_next)
        q_val_next = q_next_target.gather(1, q_next_train.argmax(1).unsqueeze(1)).squeeze(1)#q_next_target[range(batch_size),q_next_train.argmax(1)]
        q_val_target = r + gamma * q_val_next * (1 - done)
        
    q_pred = Q_train(S)
    q_pred = q_pred.gather(1, a.unsqueeze(1)).squeeze(1)#q_val[range(batch_size),a]
    
    writer.add_histogram('Predicted Q-values', q_pred, iter_number)
    writer.add_histogram('Target Q-values', q_val_target, iter_number)
    loss = (q_pred - Variable(q_val_target.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    for param in Q_train.parameters():
                param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss


# Wrapper to easy implement frame skip in envirenment
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

## Import SPACE
import os
import sys
sys.path.insert(0, '/home/rasmus/DTU/VAE_RL/SPACE/src')
from engine.utils import get_config
from model import get_model
from utils import Checkpointer, MetricLogger
from solver import get_optimizers
import os.path as osp

cfg, task = get_config()
cfg.model = 'SPACE_atari'
model = get_model(cfg)
optimizer_fg, optimizer_bg = get_optimizers(cfg, model)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg)
model.cuda()
model.eval()


class SPACE_encoder:
    def __init__(self,model):
        self.SPACE_model = model
        
    def encode(self,x):
        with torch.no_grad():
            x = torch.from_numpy(x.squeeze()).float().cuda()
            # Normalize x
            x = x/255
            
            z = self.SPACE_model.forward(x, 10000)
            # Returns space of (n_phi, H*W, 42)
            z = z.unsqueeze(0).cpu()
        
        return z
    
    
import os
if not os.path.exists('models'):
    os.makedirs('models')

#env = gym.make("PongNoFrameskip-v4")
epsilon_by_frame = lambda n_frames: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * n_frames / epsilon_decay)

env_id = "Riverraid-v0"
env = gym.make(env_id)
env = MaxAndSkipEnv(env, skip=4)
n_phi = 4 # number of frames to stack
im_dim = (256, 42)
Q_train = CnnDQN_VAE((n_phi,) + im_dim, env.action_space.n, max_pool=True)
Q_target = CnnDQN_VAE((n_phi,) + im_dim, env.action_space.n, max_pool=True)
Q_target.load_state_dict(Q_train.state_dict())

if USE_CUDA:
    Q_train = Q_train.cuda()
    Q_target = Q_target.cuda()
#optimizer = optim.Adam(Q_train.parameters(), lr=0.0015)
learning_rate = 0.000025
optimizer = optim.RMSprop(Q_train.parameters(), lr=learning_rate, alpha=0.95,momentum = 0.95)
## Saving parameters
save_path = "models/DQN_"+env_id
save_freq = 10**5
tensorboard_update_freq = 10

## Runtime hyperparamters
replay_initial = 50000
replay_buffer_size = 100000
replay_buffer = Replay_buffer(replay_buffer_size, im_dim, n_phi, S_dtype="float")
VAE_encoder = SPACE_encoder(model)

epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 10**6

final_frame = 10**9
batch_size = 32
gamma      = 0.99
## Update parameters
update_freq = 4*n_phi
target_update_freq = 30000

## Add hyper parameters to tensorboard
writer.add_text("Environment", str(env_id))
writer.add_text("Buffer size", str(replay_buffer_size))
writer.add_text("Batch size", str(batch_size))
writer.add_text("Learning rate", str(learning_rate))
writer.add_text("Frames skip", str(4))
writer.add_text("Type", "SPACE")

episode_reward = 0
episode_index = 0
par_updates = 0
n_frames = 0
while n_frames<final_frame:
    done = False
    episode_index += 1
    S = np.zeros((n_phi,) + (210,160,3), dtype="uint8")
    S_obs = np.zeros((n_phi,) + (210,160,3), dtype="uint8")
    S[n_phi-1] = env.reset()
    S = phi_transformer(S, n_phi, im_size=[128,128], n_channels = 3)
    S = VAE_encoder.encode(S)
    episode_reward = 0
    
    while not done:
        # Get random chance
        epsilon = epsilon_by_frame(n_frames)
        # Select action
        if np.random.rand(1)[0]<epsilon: # Case ranom move selected
                a = np.random.randint(env.action_space.n)
        else:
            with torch.no_grad():# Case non-random move selected greedely
                a = Q_train.act(S)

        # Advance state
        r = 0
        for j in range(n_phi):
            n_frames += 1
            S_obs[j], r_temp, done, info = env.step(a)
            r += r_temp
            episode_reward += r_temp

            if (done): # Check if game done
                # Stich missing frames together
                S_obs[(n_phi-j):n_phi] = S_obs[0:j]
                S_obs[0:(n_phi-j)] = S_obs_prev[j:n_phi]
                break
            
        S_obs_prev = S_obs.copy()
        S_next = phi_transformer(S_obs, n_phi, im_size=[128,128], n_channels = 3) # Transform input
        S_next = VAE_encoder.encode(S_next)
        n_frames += 1
        replay_buffer.add_replay([S, a, r, S_next, done])

        S = S_next

        if (n_frames > replay_initial) & (n_frames % update_freq == 0):
            par_updates += 1
            loss = compute_td_loss(batch_size,n_frames)
            if (par_updates % tensorboard_update_freq == 0):
                writer.add_scalar('loss', loss.item(), n_frames)

        if (par_updates % target_update_freq == 0):
                    Q_target.load_state_dict(Q_train.state_dict())
                
        if (n_frames % save_freq == 0):
            torch.save(Q_train, save_path)
    # Episode finished
    print(episode_reward)
    writer.add_scalar('Episode reward', episode_reward, episode_index)
