import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import gym
import sys
import yaml
import copy
import collections
from envs.config import get_environment_config
import time
import copy
import pickle


def observe(env,replay_buffer, observation_steps):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for

    """

    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def observe_population(env,replay_buffer, observation_steps, env_name):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for

    """

    env_config = get_environment_config(env_name)
    parameters = dict(zip(env_config['covariates'], np.array([np.random.uniform(r[0], r[1]) for r in env_config['train_env_range']])))
    env = env_config['env'](**parameters)
    time_steps = 0
    obs = env.reset()
    done = False
    env_config = get_environment_config(env_name)
    

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            parameters = dict(zip(env_config['covariates'], np.array([np.random.uniform(r[0], r[1]) for r in env_config['train_env_range']])))
            env = env_config['env'](**parameters)
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()




# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""

    def __init__(self, device, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.device = device

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s,  s_,a, r, d = self.storage[i]
            if not isinstance(a, collections.abc.Iterable): 
                a = [a]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array([r], copy=False))
            dones.append(np.array([d], copy=False))
        return (
                torch.FloatTensor(states).to(self.device),
                torch.FloatTensor(next_states).to(self.device),
                torch.LongTensor(actions).to(self.device),
                torch.reshape(torch.FloatTensor(rewards), (-1,1)).to(self.device),
                torch.reshape(torch.FloatTensor(dones), (-1,1)).to(self.device)
            )


    def sample_one(self):
        ind = np.random.randint(0, len(self.storage))
        return self.storage[ind]

    def save(self, save_folder):
        np.save(f'{save_folder}_buffer.npy', self.storage)
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.storage = np.load(f'{save_folder}_buffer.npy', allow_pickle=True)
        self.ptr = np.load(f"{save_folder}_ptr.npy", allow_pickle=True)
        print(len(self.storage))

def evaluate_policy(policy, env, eval_episodes=100):
    """run several episodes using the best agent policy
        
        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
           
        Returns:
            avg_reward (float): average reward over the number of evaluations
    
    """
    
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

def generate_random_environment(env_name):
    env_config = get_environment_config(env_name)
    parameters = dict(zip(env_config['covariates'], np.array([np.random.uniform(r[0], r[1]) for r in env_config['train_env_range']])))
    env = env_config['env'](**parameters)
    return env


class Runner():
    """Carries out the environment steps and adds experiences to memory"""
    
    def __init__(self, env, agent, replay_buffer, population=False, env_name=None):
        
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False
        self.population = population
        self.env_name = env_name
        
    def next_step(self, episode_timesteps, max_timesteps = 1000, noise=0.1):
        
        action = self.agent.select_action(np.array(self.obs), noise=noise)
        
        # Perform action
        new_obs, reward, done, info = self.env.step(action) 
        done_bool =  float(done)
        
        # Store data in replay buffer
        self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool))
        
        self.obs = new_obs
        
        if done:
            if self.population:
                self.env = generate_random_environment(self.env_name)
            self.obs = self.env.reset()
            done = False
            
            return reward, True
        
        return reward, done


def train(agent, test_env, runner, replay_buffer, exploration, reward_threshold, filename ='policy',directory = 'policies', eval_frequency = 50 ):
    """Train the agent for exploration steps
    
        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            exploration (int): how many training steps to run
    
    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False 
    obs = test_env.reset()
    evaluations = []
    rewards = []
    best_avg = -2000
    
    for total_timesteps in range(exploration):
    # for total_timesteps in tqdm(range(exploration)):
        if done: 
            if total_timesteps != 0: 
                rewards.append(episode_reward)
                avg_reward = np.mean(rewards[-100:])
                
                
                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save(filename,directory)
                
                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                        total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()


                
                agent.train(replay_buffer,episode_timesteps)

                
                # Evaluate episode
                if timesteps_since_eval >= eval_frequency:
                    timesteps_since_eval %= eval_frequency
                    eval_reward = evaluate_policy(agent, test_env)
                    evaluations.append(eval_reward)

                    if best_avg < eval_reward:
                        best_avg = eval_reward
                        print("saving best model....\n")
                        agent.save(filename,directory)

                        if eval_reward >= reward_threshold:
                            print(f"acheived the wanted reward of {reward_threshold}")
                            break
                    

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

        reward, done = runner.next_step(episode_timesteps, noise = 0.1 )
        episode_reward += reward
        episode_timesteps += 1
        timesteps_since_eval += 1
    agent.load(filename,directory)

    return agent

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state

