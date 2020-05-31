#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:48:30 2020

@author: tirsgaard
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
#from baseline.baselines.baselines.common.atari_wrappers import wrap_deepmind
from stable_baselines.common.atari_wrappers import wrap_deepmind
from DQN_model import CnnDQN
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.benchmark=True
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

writer = SummaryWriter()



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
    
    #writer.add_histogram('Predicted Q-values', q_pred, iter_number)
    #writer.add_histogram('Target Q-values', q_val_target, iter_number)
    loss = (q_pred - Variable(q_val_target.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    for param in Q_train.parameters():
                param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss

# Wrapper to easy implement frame skip in envirenment
class SkipFramesEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip       = skip
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

epsilon_by_frame = lambda n_frames: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * n_frames / epsilon_decay)


import os
if not os.path.exists('models'):
    os.makedirs('models')

#env = gym.make("PongNoFrameskip-v4")
env_id = "SpaceInvaders-v0"
#env_id = "Riverraid-v0"
env = gym.make(env_id)
env = wrap_deepmind(env, clip_rewards=False, frame_stack=True, episode_life=True) # We clip rewards in loss calculation
n_phi = 4 # number of frames to stack
env = SkipFramesEnv(env, skip=n_phi)
im_dim = (84,84)
Q_train = CnnDQN((n_phi, 84, 84), env.action_space.n)
Q_target = CnnDQN((n_phi, 84, 84), env.action_space.n)
Q_target.load_state_dict(Q_train.state_dict())

if USE_CUDA:
    Q_train = Q_train.cuda()
    Q_target = Q_target.cuda()
#optimizer = optim.Adam(Q_train.parameters(), lr=0.0000025)

lr_rate = 0.00025
optimizer = optim.RMSprop(Q_train.parameters(), lr = lr_rate, alpha=0.95, momentum = 0, eps = 0.01)
## Saving parameters
save_path = "models/DQN_"+env_id
save_freq = 10**5
tensorboard_update_freq = 100

## Runtime hyperparamters
replay_initial = 50000
replay_buffer_size = 10**6
replay_buffer = Replay_buffer(replay_buffer_size, im_dim, n_phi, save_type="disk")

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10**6

final_frame = 10**9
batch_size = 32
gamma      = 0.99
## Update parameters
update_freq = 4*n_phi
target_update_freq = 30000

episode_reward = 0
episode_index = 0
par_updates = 0
n_frames = 0
loss_list = []
episode_reward = 0

## Add hyper parameters to tensorboard
writer.add_text("Environment", str(env_id))
writer.add_text("Buffer size", str(replay_buffer_size))
writer.add_text("Batch size", str(batch_size))
writer.add_text("Learning rate", str(lr_rate))
writer.add_text("Frames skip", str(4))
writer.add_text("Type", "DDQN")



while n_frames<final_frame:
    done = False
    episode_index += 1
    S = np.zeros((n_phi,) + (84,84,1), dtype="uint8")
    S = env.reset()
    
    while not done:
        # Get random chance
        epsilon = epsilon_by_frame(n_frames)
        # Select action
        if np.random.rand(1)[0]<epsilon: # Case random move selected
            a = np.random.randint(env.action_space.n)
        else:
            with torch.no_grad():# Case non-random move selected greedely
                S_input = np.transpose(np.array(S), (2, 0, 1)) # Convert lazt frames to numpy
                a = Q_train.act(S_input)

        # Advance state
        S_next, r, done, info = env.step(a)
        replay_buffer.add_replay([S, a, r, S_next, done])
        S = S_next
        episode_reward += r
        n_frames += 4
        
        if (n_frames > replay_initial) & (n_frames % update_freq == 0):
            # Update model
            par_updates += 1
            loss = compute_td_loss(batch_size,n_frames)
            loss_list.append(loss.item())
            if (par_updates % tensorboard_update_freq == 0):
                writer.add_scalar('loss', np.mean(loss_list), n_frames)
                loss_list = []

        if (par_updates % target_update_freq == 0):
            # Update target network
                    Q_target.load_state_dict(Q_train.state_dict())
                
        if (n_frames % save_freq == 0):
            # Save model
            torch.save(Q_train, save_path)
    # Episode finished
    if env.was_real_done:
        print(episode_reward)
        writer.add_scalar('Episode reward', episode_reward, episode_index)
        episode_reward = 0
