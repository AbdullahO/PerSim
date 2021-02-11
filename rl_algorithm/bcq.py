'''
Implements Batch-Constrained Deep Q-Learning (BCQ)
as in https://github.com/sfujim/BCQ
'''

# in particular, Discrete BCQ

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
from utils.policy_utils import ReplayBuffer, evaluate_policy

# FC_Q has more layers than in dqn.py

class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, 256)
        self.q2 = nn.Linear(256, 256)
        self.q3 = nn.Linear(256, num_actions)

        self.i1 = nn.Linear(state_dim, 256)
        self.i2 = nn.Linear(256, 256)
        self.i3 = nn.Linear(256, num_actions)


    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i

class BCQ(object):
    def __init__(
        self,
        num_actions,
        state_dim,
        device,
        BCQ_threshold=0.3,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={"lr": 1e-5},
        polyak_target_update=True,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps = 0.9,
        end_eps = 0.1,
        eps_decay_period = 25e3,
        eval_eps=0.001,
        batch_size =128 
    ):
        self.batch_size = batch_size
        self.device = device
        self.threshold = BCQ_threshold
        # Determine network type
        self.Q = FC_Q(state_dim, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > self.eval_eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                q, imt, i = self.Q(state)
                imt = imt.exp()
                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
                # Use large negative number to mask actions from argmax
                return int((imt * q + (1. - imt) * -1e8).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train(self, replay_buffer, it):
        # Sample replay buffer
        state, next_state,action, reward, done = replay_buffer.sample(self.batch_size)

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()
        if it % 10000 == 0:
            print(Q_loss)
        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename, directory):
        torch.save(self.Q.state_dict(), f'{directory}/{filename}_Q.pth')
        torch.save(self.Q_optimizer.state_dict(), f'{directory}/{filename}_optimizer.pth')

    def load(self, filename):
        self.Q.load_state_dict(torch.load(f'{directory}/{filename}_Q.pth'))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(f'{directory}/{filename}_optimizer.pth'))


def BCQ_trainer(env, device, filename, save_dir, eval_freq, buffer_name, max_timesteps=int(1e6)):
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy = BCQ(num_actions, 
                 state_dim,
                 device)
                 

    replay_buffer = ReplayBuffer(device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    best_so_far = -np.inf

    for training_iter in tqdm(range(0, max_timesteps, eval_freq)):
        for it  in tqdm(range(eval_freq)):
            policy.train(replay_buffer, it)
        score = evaluate_policy(policy, env)
        evaluations.append(score)
        np.save(f'{save_dir}/{filename}_evals', evaluations)
        policy.save(filename, save_dir)
        best_so_far = max(best_so_far, score)
        print(f'Training iterations: {training_iter}\nEvaluations: {evaluations[-1]}\nBest Score: {best_so_far}')
