import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from utils.policy_utils import ReplayBuffer, evaluate_policy
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)


class FC_Q(torch.nn.Module):
  def __init__(self, state_dim, num_actions):
    super(FC_Q, self).__init__()
    self.l1 = nn.Linear(state_dim, 256)
    self.l2 = nn.Linear(256, 256)
    self.l3 = nn.Linear(256, num_actions)


  def forward(self, state):
    q = F.relu(self.l1(state))
    q = F.relu(self.l2(q))
    return self.l3(q)

class DQN(object):
  def __init__(
    self, 
    num_actions,
    state_dim,
    device,
    discount=0.99,
    optimizer="Adam",
    optimizer_parameters={"lr": 0.001},
    polyak_target_update=True,
    target_update_frequency=8e3,
    tau=0.005,
    initial_eps = 0.9,
    end_eps = 0.1,
    eps_decay_period = 25e3,
    eval_eps=0.001,
    batch_size = 128
  ):
    self.batch_size = batch_size
    self.device = device

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
    eps = self.eval_eps if eval \
      else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

    # Select action according to policy with probability (1-eps)
    # otherwise, select random action
    if np.random.uniform(0,1) > eps:
      with torch.no_grad():
        state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
        return int(self.Q(state).argmax(1))
    else:
      return np.random.randint(self.num_actions)


  def train(self, replay_buffer, iterations):
    for it in range(iterations):
      # Sample replay buffer
      state, next_state, action, reward, done = replay_buffer.sample(self.batch_size)
      action = action.reshape(-1,1)
      not_done = 1-done

      # Compute the target Q value
      with torch.no_grad():
        target_Q = reward + not_done * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

      # Get current Q estimate
      current_Q = (self.Q(state)).gather(1, action)

      # Compute Q loss
      Q_loss = F.smooth_l1_loss(current_Q, target_Q)

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
    torch.save(self.Q.state_dict(),  '%s/%s_Q.pth' % (directory, filename))
    torch.save(self.Q_optimizer.state_dict(), '%s/%s_optimizer.pth' % (directory, filename))
 


  def load(self, filename, directory):
    self.Q.load_state_dict(torch.load(f'{directory}/{filename}'+ "_Q.pth", map_location = self.device))
    self.Q_target = copy.deepcopy(self.Q)
    self.Q_optimizer.load_state_dict(torch.load(f'{directory}/{filename}'+ "_optimizer.pth", map_location = self.device))
    self.initial_eps = 0.01
    self.end_eps = 0.01
    



def DQN_trainer(env, device, exploration, threshold, filename ,save_dir , eval_frequency, observations, train_freq = 4, init_policy = None):

  replay_buffer = ReplayBuffer(device, max_size = 512)
  # For saving files
  setting = f"{filename}"

  state_dim = env.observation_space.shape[0]
  num_actions = env.action_space.n
    
  # Initialize and load policy
  
  if init_policy is None:
      policy = DQN(
    num_actions,
    state_dim,
    device)

  else:
        policy = init_policy

  evaluations = []

  state, done = env.reset(), False
  episode_start = True
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0
  episodes_reward = []
  best_policy_eval = -float('inf')
  # Interact with the environment for max_timesteps
  for t in tqdm(range(exploration)):

    episode_timesteps += 1
    
    if t < observations:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(state))

    # Perform action and log results
    next_state, reward, done, info = env.step(action)
    episode_reward += reward

    # # Only consider "done" if episode terminates due to failure condition
    # done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

    # Store data in replay buffer
    replay_buffer.add((state, next_state, action, reward, done))
    state = copy.copy(next_state)
    episode_start = False

    # Train agent after collecting sufficient data
    if t >= observations and (t+1) % train_freq == 0:
      policy.train(replay_buffer, 1)

    if done:
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      # Reset environment
      episodes_reward.append(episode_reward)
    
      state, done = env.reset(), False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1
      
      if episode_num%100 == 0:
         print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} average Reward (last 100) : {np.mean(episodes_reward[-100:]):.3f}")
        
    # Evaluate episode
    if  (t + 1) % eval_frequency == 0:
      evaluations.append(evaluate_policy(policy, env, 500))
      average_eval = evaluations[-1]
      if average_eval> best_policy_eval:
        policy.save(filename, save_dir)
        print(f'saving model, eval reward:{average_eval}, old reward:{best_policy_eval}')
        best_policy_eval = average_eval
    if best_policy_eval > threshold:
      print(f'acheived the wanted reward of {threshold}')
      break
  # Save final policy
  evaluations.append(evaluate_policy(policy, env))
  average_eval = evaluations[-1]
  if average_eval> best_policy_eval:
          policy.save(filename, save_dir)
          print(f'saving model, eval reward:{average_eval}, old reward:{best_policy_eval}')
  policy.load(filename, save_dir)
  return policy  
