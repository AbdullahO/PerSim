import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import gym
import sys
import copy
import collections
from envs.config import get_environment_config
import time
import copy
import pickle
from rl_algorithm.td3 import TD3
from rl_algorithm.dqn import DQN


def load_samples(path):
    with open(path, "rb") as fp:   # Unpickling
        paths = pickle.load(fp)
    return paths



def _get_empty_running_paths_dict():
    return dict(observations=[], next_observations =[], actions=[], rewards=[], dones=[])

def load_policy(env, policy_directory,policy_name, policy_class, device):
    if policy_class == 'DQN':
        state_dim = env.observation_space.shape[0]
        num_actions = env.action_space.n
        # Initialize and load policy
        policy = DQN(num_actions, state_dim, device)
        policy.load(policy_name, policy_directory)

    elif policy_class == 'TD3':
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        policy = TD3(state_dim, action_dim, max_action, env, device)
        policy.load(policy_name, policy_directory)

    else:
        raise ValueError ("policy_class should be TD3 or DQN")

    return policy