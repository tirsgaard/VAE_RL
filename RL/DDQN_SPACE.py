#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:48:30 2020

@author: tirsgaard
# This is still outdated compared to the DDQN.py
"""
import faulthandler; faulthandler.enable()
import math, random

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from collections import deque
import gym
from time import time
import glob
import pickle
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from stable_baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv

import matplotlib.pyplot as plt
from helper_functions_new import Replay_buffer
from DQN_model import CnnDDQN_VAE
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.benchmark=True
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

writer = SummaryWriter()


print("test")
## Back propergation
def compute_td_loss(batch_size, iter_number):
    S, a, r, S_next, done = replay_buffer.return_batch(batch_size)
    S      = Variable(torch.FloatTensor(np.float32(S)))
    S_next = Variable(torch.FloatTensor(np.float32(S_next)))
    a      = Variable(torch.LongTensor(a))
    r      = torch.sign(Variable(torch.FloatTensor(r)))
    done   = Variable(torch.FloatTensor(done))

    with torch.no_grad():
        # Normalize
        S = normalizer.normalize(S)
        S_next = normalizer.normalize(S_next)
        
        q_next_train = Q_train(S_next)
        q_next_target = Q_target(S_next)
        q_val_next = q_next_target.gather(1, q_next_train.argmax(1).unsqueeze(1)).squeeze(1)#q_next_target[range(batch_size),q_next_train.argmax(1)]
        q_val_target = r + gamma * q_val_next * (1 - done)
        
    q_pred = Q_train(S)
    q_pred = q_pred.gather(1, a.unsqueeze(1)).squeeze(1)#q_val[range(batch_size),a]
    
    loss_func = nn.SmoothL1Loss()
    loss = loss_func(q_pred, q_val_target)
    optimizer.zero_grad()
    loss.backward()
    #for param in Q_train.parameters():
    #            param.grad.data.clamp_(-1, 1)
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
    
class MaxEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        self.obs, reward, done, info = self.env.step(action)
        self.old_obs = self.obs # we store old obs to max_pool
        
        # Max pool frames
        obs = np.maximum(self.obs, self.old_obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.old_frame = self.env.reset(**kwargs)
        return self.old_frame
    
class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs
    
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        return self.transform(frame)

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return list(self.frames)
    
    
def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """
    Configure environment for DeepMind-style Atari.
    :param env: (Gym Environment) the atari environment
    :param episode_life: (bool) wrap the episode life wrapper
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :param frame_stack: (bool) wrap the frame stacking wrapper
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped atari environment
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

class Normalizer():
    def __init__(self, num_inputs):
        self.mean = torch.zeros(num_inputs).cuda()
        self.var = torch.zeros(num_inputs).cuda()
        self.n = torch.zeros(1).cuda()
        
    def collect_and_norm(self, x):
        ### Assumes input format is (1,C,H,W)
        # Update values
        n_samples = x.shape[2]*x.shape[3]
        old_n = self.n.clone()
        self.n += n_samples
        new_mean = x.mean(dim=(0,2,3))
        new_var = x.var(dim=(0,2,3))
        
        self.mean = self.mean*old_n/self.n + new_mean*n_samples/self.n
        self.var = self.var*old_n/self.n + new_var*n_samples/self.n
        
        # Return normalized image
        obs_std = torch.sqrt(self.var)
        return ((x.permute(0,2,3,1) - self.mean)/obs_std).permute(0,3,1,2)

    def normalize(self, inputs):
        ### Assumes input format is (B,C,H,W)
        obs_std = torch.sqrt(self.var)
        return ((inputs.permute(0,2,3,1)  - self.mean)/obs_std).permute(0,3,1,2)
    
def proc_S(S):
    # Convert from torch dim=(1,C,H,W) to numpy = (H,W,C)
    S = S.squeeze(0)
    S = S.permute(1,2,0)
    S = S.cpu().numpy()
    return S

epsilon_by_frame = lambda n_frames: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * n_frames / epsilon_decay)


## Import SPACE
import os
import sys
sys.path.append(os.path.abspath('../../SPACE_own/src')) # location of space files
from engine.utils import get_config
from model import get_model
from utils import Checkpointer, MetricLogger
from solver import get_optimizers
import os.path as osp

cfg, task = get_config()
cfg.resume_ckpt = os.path.abspath('VAE_models/SPACE/FishingDerby/model_000060001.pth') # Location of space model
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
            x = torch.stack(x).cuda()
            
            z = self.SPACE_model.forward(x, 10000)
            # Returns space of (n_phi, H*W, 42)
            z = z.unsqueeze(0).cpu()
        
        return z
    
    
import os
if not os.path.exists('models'):
    os.makedirs('models')

env_id = "FishingDerby-v0"
env = gym.make(env_id)
env = MaxEnv(env)
env = wrap_deepmind(env, clip_rewards=False, frame_stack=True, episode_life=True)# We clip rewards in loss calculation
env = SkipEnv(env) 

n_phi = 168 # number of frames to stack
im_dim = (16, 16)
Q_train = CnnDDQN_VAE((n_phi,) + im_dim, env.action_space.n, max_pool=False)
Q_target = CnnDDQN_VAE((n_phi,) + im_dim, 
                       env.action_space.n, 
                       max_pool=False, 
                       prov_bias = Q_train.return_bias())
Q_target.load_state_dict(Q_train.state_dict())

if USE_CUDA:
    Q_train = Q_train.cuda()
    Q_target = Q_target.cuda()

lr_rate = 0.0000625
optimizer = optim.Adam(Q_train.parameters(), lr=lr_rate)
## Saving parameters
save_path = "models/DDQN_SPACE_"+env_id
back_up_path = "/work3/s174511/backup/DDQN_SPACE_"+env_id+"/"
save_freq = 10**5
tensorboard_update_freq = 100
resume = False
run_time = 23*3600 # in seconds

## Runtime hyperparamters
notes = "Fishing Derby SPACE model" # If anything should be noted in tensorboard

## Runtime hyperparamters
replay_initial = 50000
replay_buffer_size = 10**6
VAE_encoder = SPACE_encoder(model)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10**6

final_frame = 10**9
batch_size = 32
gamma      = 0.99
## Update parameters
update_freq = 4
target_update_freq = 3*10**4 # following Double Q-learing
normalizer = Normalizer((4*42))


episode_reward = 0
episode_index = 0
par_updates = 0
n_frames = 0
loss_list = []
fps_list = []
episode_reward = 0
start_time = time()

if resume:
    ## Loading newest tensorboard
    list_of_files = glob.glob('runs/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    writer = SummaryWriter(latest_file)
    
    ## Load ANN models
    # find newest backup
    Q_target.load_state_dict(torch.load(back_up_path+"Q_target"))
    Q_train.load_state_dict(torch.load(back_up_path+"Q_train"))
    
    replay_buffer = Replay_buffer(replay_buffer_size, im_dim, n_phi, save_type="disk", S_dtype="float16", cache=False, resume=True, save_location = back_up_path)
    # Load last time step
    with open(back_up_path+'objs.pkl', 'rb') as f:
        n_frames, episode_index = pickle.load(f)
    
    
else:
    writer = SummaryWriter()
    ## Add hyper parameters to tensorboard
    writer.add_text("Environment", str(env_id))
    writer.add_text("Buffer size", str(replay_buffer_size))
    writer.add_text("Batch size", str(batch_size))
    writer.add_text("Learning rate", str(lr_rate))
    writer.add_text("Frames skip", str(4))
    writer.add_text("Type", "SPACE")
    writer.add_text("Notes", notes)
    
    replay_buffer = Replay_buffer(replay_buffer_size, im_dim, n_phi, save_type="disk", S_dtype="float16", cache=False, save_location = back_up_path)



t1 = time()
while n_frames<final_frame:
    done = False
    episode_index += 1
    S = env.reset()
    print(torch.stack(S).shape)
    with torch.no_grad():
        S = VAE_encoder.encode(S)
        S = S.permute((0,1,3,2))
        S = S.reshape(1, 4*42, 16, 16)
    
    while not done:
        # Get fps
        t2 = time()
        fps_list.append(1/(t2-t1))
        t1 = t2
        # Get random chance
        epsilon = epsilon_by_frame(n_frames)
        # Select action
        with torch.no_grad():
            S_normed = normalizer.collect_and_norm(S.cuda()) # normalize
        if np.random.rand(1)[0]<epsilon: # Case ranom move selected
                a = np.random.randint(env.action_space.n)
        else:
            with torch.no_grad():# Case non-random move selected greedely
                a = Q_train.act(S_normed)

        # Advance state
        S_next, r, done, info = env.step(a)
        with torch.no_grad():
            S_next = VAE_encoder.encode(S_next)
            S_next = S_next.permute((0,1,3,2))
            S_next = S_next.reshape(1, 4*42, 16, 16)
            replay_buffer.add_replay([proc_S(S), a, r, proc_S(S_next), done])
            
        S = S_next
        episode_reward += r
        n_frames += 1

        if (n_frames > replay_initial) & (n_frames % update_freq == 0):
            # Update model
            loss = compute_td_loss(batch_size,n_frames)
            loss_list.append(loss.item())
            if (n_frames % tensorboard_update_freq == 0):
                writer.add_scalar('loss', np.mean(loss_list), n_frames)
                loss_list = []

        if (n_frames % target_update_freq == 0): # This would instead be based on parameter updates following the nature paper, but in their code it is based on frames
            # Update target network
                Q_target.load_state_dict(Q_train.state_dict())
                
        if (n_frames % save_freq == 0):
            # Save model
            torch.save(Q_train, save_path)
    # Episode finished
    if env.was_real_done:
        print(episode_reward)
        writer.add_scalar('Episode reward', episode_reward, episode_index)
        writer.add_scalar('FPS', np.array(fps_list).mean(), episode_index)
        fps_list = []
        episode_reward = 0
        
        # Check if it is time stop
        total_time = time()-start_time
        if (total_time>=run_time):
            # Save everything
            torch.save(Q_target.state_dict(), back_up_path + "Q_target")
            torch.save(Q_train.state_dict(), back_up_path + "Q_train")
            replay_buffer.save_buffer()
            with open(back_up_path+'objs.pkl', 'wb') as f:
                pickle.dump([n_frames, episode_index], f)
            # Stop program
            print("Program stopped")
            break
