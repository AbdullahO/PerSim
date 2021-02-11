import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import evaluate_policy
from envs.config import get_environment_config
import sys
import yaml

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477    
   
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
            

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device#torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        env,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{directory}/{filename}_critic_optimizer.pth")
            
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{directory}/{filename}_actor_optimizer.pth")


    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(f"{directory}/{filename}_critic_optimizer.pth", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth",map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(f"{directory}/{filename}_actor_optimizer.pth",map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)


def TD3_trainer(env, device, exploration, threshold, filename ,save_dir , eval_frequency, observations, init_policy = None, population=False, env_name='halfCheetah'):
    print("population: ", population)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # adjustable parameters
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    discount = 0.99
    tau = 0.005
    max_timesteps = exploration
    start_timesteps = observations
    batch_size = 256
    expl_noise = 0.1

    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
	

    kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": discount,
            "tau": tau,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq
    policy = TD3(env = env, device = device, **kwargs)
    
    # if load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim, device)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy, env)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_avg = -2000
    episode_rewards = []

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_timesteps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done:
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            if best_avg < avg_reward:
                best_avg = avg_reward
                print("Saving best model.... \n")
                policy.save(filename, save_dir)
                
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0 
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_frequency == 0:
            evaluations.append(evaluate_policy(policy, env))
            np.save(f"eval/{filename}", evaluations)
            if evaluations[-1] > threshold:
                print('reached wanted reward')
                break
    #env_config = get_environment_config(env_name)
    if population:
        env_config = get_environment_config(env_name)
        print("------------------------------------------------")
        print("STARTING ONLINE POPULATION EVALUATION")
        for covariates in env_config['test_env']:
            parameters = dict(zip(env_config['covariates'], covariates))
            test_env = env_config['env'](**parameters)
            average_eval = evaluate_policy(policy, test_env)
            print(parameters, average_eval)
        print("------------------------------------------------")
    else:
        print("------------------------------------------------")
        print("FINAL AGENT EVAL: ", evaluate_policy(policy, env))
        print("------------------------------------------------")
    return policy


def TD3_trainer_sim(env, sim, test_unit, device, exploration, threshold, filename ,save_dir , eval_frequency, observations, init_policy = None, population=False, env_name=None):
    print("population: ", population)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # adjustable parameters
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    discount = 0.99
    tau = 0.005
    max_timesteps = 1e6
    start_timesteps = 25e3
    batch_size = 256
    expl_noise = 0.1

    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    reward_fn = env.torch_reward_fn()

    # Set seeds
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    

    kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": discount,
            "tau": tau,
    }

    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq
    policy = TD3(env = env, device = device, **kwargs)
    
    # if load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy, env)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_avg = -2000
    episode_rewards = []

    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        
        # pre process
        state, action = torch.tensor(state).to(device).float(), torch.tensor(action).to(device).float()
        sim.reset(state)
        # print(action)
        # Perform action
        next_state, _, _, _ = sim.step(action, test_unit)
        
        # post process predictions
        next_state = next_state.detach()
        reward = reward_fn(state, action, next_state)
        done_bool = 1.0 if episode_timesteps < env.max_timesteps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done:
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            if best_avg < avg_reward:
                best_avg = avg_reward
                print("Saving best model.... \n")
                policy.save(filename, save_dir)

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0 
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_frequency == 0:
            evaluations.append(evaluate_policy(policy, env))
            np.save(f"eval/{filename}", evaluations)

    print("------------------------------------------------")
    print("FINAL AGENT EVAL: ", evaluate_policy(policy, env))
    print("------------------------------------------------")
    return policy




