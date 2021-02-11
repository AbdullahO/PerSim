import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import gym

from envs.config import get_environment_config
from utils.policy_utils import evaluate_policy
from rl_algorithm.sacd.model import TwinnedQNetwork, CateoricalPolicy, FlatTwinnedQNetwork, FlatCateoricalPolicy
from rl_algorithm.sacd.utils import disable_gradients, update_params, RunningMeanStats
from rl_algorithm.sacd.memory import (
    LazyMultiStepMemory,
    LazyPrioritizedMultiStepMemory,
    FlatMultiStepMemory
)

class BaseAgent(ABC):

    def __init__(self, env, test_env, device, filename, log_dir, num_steps=100000, batch_size=64,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0):
        super().__init__()

        self.env = env
        self.test_env = test_env

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = device
        
        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            if len(self.env.observation_space.shape) == 3:
                self.memory = LazyPrioritizedMultiStepMemory(
                    capacity=memory_size,
                    state_shape=self.env.observation_space.shape,
                    device=self.device, gamma=gamma, multi_step=multi_step,
                    beta_steps=beta_steps)
            else:
                raise NotImplementedError
        else:
            if len(self.env.observation_space.shape) == 3:
                self.memory = LazyMultiStepMemory(
                    capacity=memory_size,
                    state_shape=self.env.observation_space.shape,
                    device=self.device, gamma=gamma, multi_step=multi_step)
            else:
                self.memory = FlatMultiStepMemory(
                    capacity=memory_size,
                    state_shape=self.env.observation_space.shape,
                    device=self.device, gamma=gamma, multi_step=multi_step)

        self.filename = filename
        self.log_dir = log_dir
        self.train_return = RunningMeanStats(log_interval)
        self.evaluations = []
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_policy_eval = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            self.train_episode()
            if self.episodes % 100 == 0:
                print(self.steps, "/", self.num_steps)
            if self.steps > self.num_steps:
                break
        np.save(f"eval/{self.filename}.npy", self.evaluations)

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
            else:
                action = self.explore(state)

            next_state, reward, done, _ = self.env.step(action)

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)

            # To calculate efficiently, set priority=max_priority here.
            self.memory.append(state, action, clipped_reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            if self.is_update():
                self.train()

            if self.steps % self.target_update_interval == 0:
                self.update_target()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                # self.save(os.path.join(self.model_dir, 'final'))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            print(f'Episode: {self.episodes:<4}  '
                f'Episode steps: {episode_steps:<4}  '
                f'Return: {episode_return:<5.1f}')

    def train(self):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and\
            hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            #print("starting next eval")
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                action = self.exploit(state)
                #if self.test_env.steps_beyond_done == 0:
                #    print(f"num_episodes: {num_episodes} steps beyond done: {self.test_env.steps_beyond_done}")
                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes
        self.evaluations.append(mean_return)

        if mean_return > self.best_policy_eval:
            print(f'saving model, eval reward:{mean_return}, old reward:{self.best_policy_eval}')
            self.best_policy_eval = mean_return
            self.save(self.filename, self.log_dir)

    
    def select_action(self, state):
        return self.exploit(state)

    @abstractmethod
    def save(self, filename, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        self.test_env.close()



class SACD(BaseAgent):

    def __init__(self, env, test_env, device, filename, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__(
            env, test_env, device, filename, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed)

        # Define networks.
        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[0], self.env.action_space.n
            ).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            dueling_net=dueling_net).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            dueling_net=dueling_net).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save(self, filename, save_dir):
        super().save(filename, save_dir)
        self.policy.save(f'{save_dir}/{filename}_policy.pth')
        self.online_critic.save(f'{save_dir}/{filename}_online_critic.pth')
        self.target_critic.save(f'{save_dir}/{filename}_target_critic.pth')
    
    def load(self, filename, save_dir):
        self.policy.load(f'{save_dir}/{filename}_policy.pth')
        self.online_critic.load(f'{save_dir}/{filename}_online_critic.pth')
        self.target_critic.load(f'{save_dir}/{filename}_target_critic.pth')
       
class FlatSACD(SACD):

    def __init__(self, env, test_env, device, filename, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0, units=[256, 256]):
        super(FlatSACD, self).__init__(
            env, test_env, device, filename, log_dir, num_steps, batch_size, lr, memory_size,
            gamma, multi_step, target_entropy_ratio, start_steps,
            update_interval, target_update_interval, False, dueling_net,
            num_eval_steps, max_episode_steps, log_interval, eval_interval,
            cuda, seed)
        # print(env, test_env, num_steps, batch_size, lr, memory_size, gamma, multi_step, target_entropy_ratio, start_steps, update_interval, target_update_interval, dueling_net, num_eval_steps, max_episode_steps, log_interval, eval_interval, cuda, seed, units)

        del self.policy
        del self.online_critic
        del self.target_critic
        del self.policy_optim
        del self.q1_optim
        del self.q2_optim

        # Define networks.
        self.policy = FlatCateoricalPolicy(
            self.env.observation_space.shape[0], self.env.action_space.n, units
            ).to(self.device)
        self.online_critic = FlatTwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            units, dueling_net=dueling_net).to(self.device)
        self.target_critic = FlatTwinnedQNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n,
            units, dueling_net=dueling_net).to(self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

    def explore(self, state):
        # Act with randomness.
        state = torch.FloatTensor(state[None, ...]).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.FloatTensor(state[None, ...]).to(self.device)
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def select_action(self, state):
        return self.exploit(state)

        
def SACD_trainer(env, device, exploration, threshold, filename, save_dir, eval_freq=1000, 
                 batch_size=64, lr=0.0003, memory_size=1000000, 
                 gamma=0.99, multi_step=1, target_entropy_ratio=0.98, 
                 start_steps=20000, update_interval=4, target_update_interval=8000,
                 dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10,
                 cuda=True, seed=0, units=[256, 256], 
                 population=False, environment=''):

 
    env_config = get_environment_config(environment)
    test_env = env
    if population:
        test_env = env_config['env']()  

    policy = FlatSACD(env, test_env, device, filename=filename, log_dir=save_dir, 
                 num_steps=exploration, batch_size=batch_size, lr=lr, 
                 memory_size=memory_size, gamma=gamma, multi_step=multi_step, 
                 target_entropy_ratio=target_entropy_ratio, 
                 start_steps=start_steps, update_interval=update_interval, 
                 target_update_interval=target_update_interval, 
                 dueling_net=dueling_net, num_eval_steps=num_eval_steps, 
                 max_episode_steps=max_episode_steps, 
                 log_interval=log_interval, eval_interval=eval_freq, 
                 units = units)
    
    # # Alternate method:
    policy.run()
    
    if population:
        f = open(f"eval/{policy.filename}.txt", "w")
        f.write("-------------------------------------")
        f.write("STARTING ONLINE POPULATION EVALUATION")
        for covariates in env_config['test_env']:
            parameters = dict(zip(env_config['covariates'], covariates))
            test_env = env_config['env'](**parameters)
            average_eval = evaluate_policy(policy, test_env)
            f.write(f"Parameters: {parameters}, Average_Eval: {average_eval}")
        f.close()

    else:
        f = open(f"eval/{policy.filename}.txt", "w")
        f.write("-----------------------------------")
        average_eval = evaluate_policy(policy, env)
        f.write(f"ONLINE AGENT EVAL: {average_eval}")
        f.close()
    
    return policy
    
    evaluations = []
    episodes_reward = []
    best_policy_eval = -float('inf')
    
    while policy.steps < policy.num_steps:
        policy.episodes += 1
        episode_reward = 0.
        episode_steps = 0

        done = False
        if population:
            parameters = dict(zip(env_config['covariates'], np.array([np.random.uniform(r[0], r[1]) for r in env_config['train_env_range']])))
            policy.env = env_config['env'](**parameters)
            policy.test_env = env_config['env'](**parameters)
            
        state = policy.env.reset()

        while (not done) and episode_steps <= policy.max_episode_steps:

            if policy.start_steps > policy.steps:
                action = policy.env.action_space.sample()
            else:
                action = policy.explore(state)

            next_state, reward, done, _ = policy.env.step(action)

            # Clip reward to [-1.0, 1.0].
            clipped_reward = max(min(reward, 1.0), -1.0)

            # To calculate efficiently, set priority=max_priority here.
            policy.memory.append(state, action, clipped_reward, next_state, done)

            policy.steps += 1
            episode_steps += 1
            episode_reward += reward
            state = next_state

            if policy.is_update():
                policy.train()

            if policy.steps % policy.target_update_interval == 0:
                policy.update_target()

            if policy.steps % policy.eval_interval == 0:
                print("Num steps: ", policy.steps)
                evaluations.append(evaluate_policy(policy, env))
                average_eval = evaluations[-1]
                if average_eval> best_policy_eval:
                    policy.save(filename, save_dir)
                    print(f'saving model, eval reward:{average_eval}, old reward:{best_policy_eval}')
                    best_policy_eval = average_eval
            if best_policy_eval > threshold:
                print(f'acheived the wanted reward of {threshold}')
                break

        # We log running mean of training rewards.
        # self.train_return.append(episode_return)

        episodes_reward.append(episode_reward)

        print(episode_reward, episode_steps)
        if policy.episodes % 10 == 0:
            print(f"Total T: {policy.steps} Episode Num: {policy.episodes} Episode T: {episode_steps} average Reward (last 100) : {np.mean(episodes_reward[-100:]):.3f}")
        
    evaluations.append(evaluate_policy(policy, env))
    average_eval = evaluations[-1]
    if average_eval > best_policy_eval:
        policy.save(filename, save_dir)
        print(f'saving model, eval reward:{average_eval}, old reward:{best_policy_eval}')
    policy.load(filename, save_dir)


    if population:
        print("-------------------------------------")
        print("STARTING ONLINE POPULATION EVALUATION")
        for covariates in env_config['test_env']:
            parameters = dict(zip(env_config['covariates'], covariates))
            test_env = env_config['env'](**parameters)
            average_eval = evaluate_policy(policy, test_env)
            print(parameters, average_eval)

    else:
        print("-----------------------------------")
        average_eval = evaluate_policy(policy, env)
        print(f"ONLINE AGENT EVAL: {average_eval}")

    return policy


