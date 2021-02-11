import torch
from simulator.simulator import Simulator
from math import ceil
import numpy as np
from tqdm import tqdm
from envs.config import get_environment_config 
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from rl_algorithm.td3 import TD3
from rl_algorithm.dqn import DQN
from utils import load_policy, load_samples
import pandas as pd 
from mpc import DynamicsFunc,MPC_M
from typing import Tuple
from torch import Tensor

class Dynamics(DynamicsFunc):
    """Exact dynamics of the pendulum.

    For this demo we assume we know the exact dynamics. These are taken from the Pendulum-v0 environment in OpenAI Gym.
    """
    def __init__(self, sim, env, discerte_action, N, unit, device ) -> Tuple[Tensor, Tensor]:
        self.env = env

        if discerte_action:  action_dim = self.env.action_space.n
        else: action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]
        env_name = sim.split('_')[0]
        rank = int(sim.split('_')[5])
        lag = int(sim.split('_')[4])
        delta = sim.split('_')[6][:5] == 'delta'
        self.delta = delta
        self.sim = Simulator(N, action_dim, state_dim,rank , device, lags= lag, state_layers = state_layers[env_name], action_layers = action_layers[env_name], continous_action = not discerte_action, delta = delta)
        self.sim.load(f'simulator/trained/{sim}')
        self.N = N
        self.reward_fun = self.env.torch_reward_fn()
        self.done_fn = self.env.torch_done_fn()
        self.unit = unit
        self.device = device

    def step(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:

        if len(actions.shape) == 1:
            actions = actions.reshape(-1,1)
        next_states = self.sim.step(states, actions, self.unit)[0]

        # print(states.shape, next_states.shape)
        objective_cost = -1*self.reward_fun(states, actions, next_states)
        dones = self.done_fn(next_states) #torch.zeros(states.shape[0], device = self.device)
        return next_states, objective_cost, dones 



def test_simulator(simulator_name, env_name, device, num_episodes, T = 50):
    if env_name in discerte_action_envs:
      discerte_action = True
    else:
      discerte_action = False

    # load env_name
    env_config = get_environment_config(env_name)
    N = env_config['number_of_units']
    env = env_config['env']
    
    if discerte_action:  action_dim = env().action_space.n
    else: action_dim = env().action_space.shape[0]

    state_dim = env().observation_space.shape[0]
    
    rank = int(simulator_name.split('_')[5])
    lag = int(simulator_name.split('_')[4])
    delta = simulator_name.split('_')[6][:5] == 'delta'
    sim = Simulator(N, action_dim, state_dim, rank, device, lags= lag, continous_action = not discerte_action, state_layers = state_layers[env_name], action_layers = action_layers[env_name], delta = delta)

    sim.load(f'simulator/trained/{simulator_name}')

    # load policy
    policy_name = f'{env_name}_{policy_class[env_name]}_test'
    policy = load_policy(env(), 'policies',policy_name , policy_class[env_name], device)

    # init parameters
    n_episodes = num_episodes
    test_covaraites  =   env_config['test_env']
    n_t = len(test_covaraites)
    data = []
    
    # init trajecotry storage
    # state_all = np.zeros([n_t, 2, n_episodes, state_dim+1, T+1])
    indices = np.arange(5)
    j = -1
    for k, unit in zip(indices,test_covaraites[:]):   
        j+=1
        MSE =[]
        r2 =[] 
        parameters = dict(zip(env_config['covariates'],unit))
        state_true = np.zeros([n_episodes, state_dim, T+1])
        state_sim = torch.zeros([n_episodes, state_dim, T+1]).to(device)
        for i_episode in tqdm(range(n_episodes)):           
            env_ = env(**parameters)
            actions = np.zeros(T+1)
            total_reward = 0
            observation = env_.reset()
            done = False
            i = 0
            while i <= T:
                # random action
                # action = env_.action_space.sample()
                # test policy action
                action = policy.select_action(np.array(observation))
                old_x0 = observation
                if not done:
                  observation, _, done,_ = env_.step(action)
                  state_true[i_episode,:,i] = observation
                  
                  if i > 0:
                      old_x0 = state_sim[i_episode, :,i-1].reshape(1,-1)
                  else:
                      old_x0 = torch.tensor(old_x0).float().reshape(1,-1).to(device)
                  action = torch.tensor([action]).reshape(1,-1).to(device)
                  # old_x0 = torch.tensor(old_x0).float().reshape(1,-1).to(device)
                  predicted_state = sim.step(old_x0, action,k)[0] 
                  # print(predicted_state.shape)
                  state_sim[i_episode, :,i] = predicted_state
                  i +=1

                else:
                  state_sim[i_episode,:,i:] = np.nan
                  state_true[i_episode,:,i:]  = np.nan
                  break
            
            mse = np.sqrt(np.nanmean(np.square(state_sim[i_episode,:,:].cpu().detach().numpy() - state_true[i_episode,:,:])))
            r2_ = r2_score(state_true[i_episode,:,:i].T, state_sim[i_episode,:,:i].cpu().detach().numpy().T , multioutput = 'variance_weighted')
            r2.append(r2_)
            MSE.append(mse)
        
        data.append([j,np.mean(MSE),np.std(MSE), np.mean(r2),np.median(r2), ])
    data = pd.DataFrame(data, columns = ['agent', 'mean', 'std', 'r2_score_mean', 'r2_score_median'])
    print('prediction error results:')
    print(data)
    data.to_csv(f'simulator/results/{simulator_name}_mse_results.csv')




def format_data(data, number_of_units, device, delta, discerte_action, action_dim, lags = 1):
  '''
  Return data formatted for pytorch.
  '''
  U = []
  I = []
  Time = []
  M = []
  Y = []
  state_dim = data[0]['observations'][0].shape[0]
  for trajectory in data[:]:
    metrics_lags = np.zeros([state_dim * lags])
    for t, (action, metrics, metrics_new) in enumerate(zip(trajectory['actions'], 
                                            trajectory['observations'], trajectory['next_observations'])):
      
      metrics_lags[state_dim:] = metrics_lags[:-state_dim]
      metrics_lags[:state_dim] = metrics
      if t+1 >= lags:
        unit = np.zeros(number_of_units)
        
        unit[trajectory['unit_info']['id']] = 1
        U.append(unit)
        if discerte_action:
          a = np.zeros(action_dim)
          a[action] = 1
        else:
          a = action
        I.append(a)
        M.append(list(metrics_lags))
        if delta: 
          Y.append(metrics_new - metrics)
        else:
          Y.append(metrics_new)
  U = np.array(U)
  I = np.array(I)
  M = np.array(M)
  Y = np.array(Y)
  if len(I.shape) == 1:
   I = I.reshape([-1,1])
  U = torch.from_numpy(U).float()
  I = torch.from_numpy(I).float()
  M = torch.from_numpy(M).float()
  Y = torch.from_numpy(Y).float()

  return U.to(device), I.to(device), M.to(device), Y.to(device)



def train(dataname,env, rank, device, delta = True, normalize_state = True, normalize_output = True, lag =1, iterations = 300, filename = None):
    # config env
    env_config = get_environment_config(env)

    ## Discrete or continous action?
    if env in discerte_action_envs:
        discerte_action = True
    else:
        discerte_action = False

    if discerte_action:  
      action_dim = env_config['env']().action_space.n
    else: 
      action_dim = env_config['env']().action_space.shape[0]

    # load data
    data_train = load_samples('datasets/'+dataname+'.pkl')
    N = env_config['number_of_units']
    u_data, i_data, m_data, y_data = format_data(data_train[:], N, device, delta, discerte_action, action_dim, lag)
    loss_fn = torch.nn.MSELoss(reduction='mean')  
    state_dim = y_data.shape[1]

    # init parameters
    means, stds, means_state, stds_state = None, None, None, None

    if normalize_state:
        means_state  = torch.mean(m_data,0)
        stds_state  = torch.std(m_data,0)

    if normalize_output:
        means  = torch.mean(y_data,0)
        stds  = torch.std(y_data,0)

    MSE = []
    delta = 'delta' if delta else 'no_delta'

    if filename is None: 
      filename = f'{dataname}_{lag}_{rank}_{delta}'

    ## train  simulator
    sim = Simulator(N, action_dim, state_dim, rank, device, lags= lag, continous_action = not discerte_action, means_state = means_state, stds_state = stds_state,means = means, stds = stds, delta =delta, state_layers = state_layers[env], action_layers = action_layers[env])
    sim.train([u_data, i_data, m_data[:,:state_dim*lag], y_data], it=iterations, learning_rate = 5e-3)
    sim.save(f'simulator/trained/{filename}')
    

def eval_policy(simulators_, env_name, num_evaluations, device):
    env_config = get_environment_config(env_name)
    if env_name in discerte_action_envs:
          discerte_action = True
    else:
          discerte_action = False

    if discerte_action:  action_dimension = env_config['env']().action_space.n
    else: action_dimension = env_config['env']().action_space.shape[0]
    
    state_dimension= env_config['env']().observation_space.shape[0]

    res = np.zeros([len(env_config['test_env'])*num_evaluations, 3 ])
    N = env_config['number_of_units']
    j = 0
    for tt, tests in enumerate(env_config['test_env'][:]):
        parameters = dict(zip(env_config['covariates'],tests))
        env = env_config['env'](**parameters)
        th =  mpc_parameters[env_name]['time_horizon']
        num_rollouts = mpc_parameters[env_name]['num_rollouts']
        num_elites = 20
        num_iterations = 10
        max_action = mpc_parameters[env_name]['max_action']
        mountainCar = env_name == 'mountainCar'
        
        simulators = [Dynamics(sim, env, discerte_action,N, tt, device) for sim in simulators_]
        mpc = MPC_M(dynamics_func=simulators, state_dimen=state_dimension, action_dimen=action_dimension,
                            time_horizon=th, num_rollouts=num_rollouts, num_elites=num_elites, 
                            num_iterations=num_iterations, disceret_action = discerte_action, mountain_car = mountainCar, max_action = max_action)
        
        sum_reward = 0
        observation = env.reset()
        k = 0
        i = 0
        while k < num_evaluations:
            state = observation

            actions, terminal_reward = mpc.get_actions(torch.tensor(state, device = device))
            action = actions[0]
            if discerte_action: 
                action = action.cpu().numpy()[0]
            else:
                action = action.cpu().numpy()
            
            observation, reward, done, info = env.step(action)
            

            sum_reward+=reward
            if mountainCar and terminal_reward < mpc._rollout_function._time_horizon -5:
               mpc._rollout_function._time_horizon = int(terminal_reward.item())+5
            
            if i%50 ==0 :
                print(f"Reward so far for agent {tt} ,  timestep {i} ,  {k}-th episode: {sum_reward}")
            i+=1
            if done:
                i = 0
                print(f"Reward for agent {tt} in the {k}-th episode: {sum_reward}")
                observation = env.reset()
                k+=1
                res[j,:] = [k, tt, sum_reward ]
                j+=1
                sum_reward = 0
                mpc = MPC_M(dynamics_func=simulators, state_dimen=state_dimension, action_dimen=action_dimension,
                            time_horizon=th, num_rollouts=num_rollouts, num_elites=num_elites, num_iterations=num_iterations, disceret_action = discerte_action, max_action = max_action, mountain_car = mountainCar)


        env.close()
    res = pd.DataFrame(res, columns = ['trial','agent', 'reward'])
    res.to_csv(f'simulator/mpc_results/mpc_sim_{simulators_[0]}.csv')


 
##### Fixed Parameters ####

action_layers = {'mountainCar':[50], 'cartPole':[50], 'halfCheetah':[256,256], 'ant':[256,256]}
state_layers = {'mountainCar':[256], 'cartPole':[256], 'halfCheetah':[256,512,512], 'ant':[256,512,512]}
policy_class = {'mountainCar':'DQN', 'cartPole':'DQN', 'halfCheetah':'TD3', 'slimHumanoid':'TD3', 'ant':'TD3'}
mpc_parameters = { 'mountainCar':
{
    'time_horizon':50, 'num_rollouts': 1000, "max_action": None
},
 'cartPole':
{
    'time_horizon':30, 'num_rollouts': 2000, "max_action": None
},
 'halfCheetah':
{
    'time_horizon':30, 'num_rollouts': 200, "max_action": 1
},
 'ant':
{
    'time_horizon':30, 'num_rollouts': 200, "max_action": 1
}
}


discerte_action_envs = {'mountainCar', 'cartPole'}
parser = argparse.ArgumentParser(description='interface of running experiments for  baselines')
parser.add_argument('--env', type=str,  default='mountainCar', help='choose envs to generate data for')
parser.add_argument('--dataname', type=str,  default='mountainCar_random_0.0_0', help='choose envs to generate data for')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--trial', type=int,  default=0, help='trial number')

parser.add_argument('--r', type=int,  default=3, help='tensor rank')
parser.add_argument('--lag', type=int,default = 1, help='lag')
parser.add_argument('--delta', dest='delta', action='store_true')
parser.add_argument('--no-delta', dest='delta', action='store_false')
parser.add_argument('--normalize_state', dest = 'normalize_state',  action='store_true')
parser.add_argument('--normalize_output', dest = 'normalize_output',  action='store_true')
parser.add_argument('--no_normalize_state', dest = 'normalize_state',  action='store_false')
parser.add_argument('--no_normalize_output', dest = 'normalize_output',  action='store_false')
parser.add_argument('--num_episodes', type=int, default=200, help='gpu device id')
parser.add_argument('--num_mpc_evals', type=int, default=20, help='number of MPC episodes')
parser.add_argument('--num_simulators', type=int, default=5, help='number of models')


parser.set_defaults(delta=True)
parser.set_defaults(normalize_state=True)
parser.set_defaults(normalize_output=True)

args = parser.parse_args()





# set device
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
delta = 'delta' if args.delta else 'no_delta'

simulators = []
# train and test simulators
for i in range(args.num_simulators):
  print('=='*10)
  print(f'Train the {i}-th simulator')
  print('=='*10)
  filename = f'{args.dataname}_{args.lag}_{args.r}_{delta}_{i}_{args.trial}'
  train(args.dataname, args.env, args.r, device, normalize_state = args.normalize_state, normalize_output = args.normalize_output, iterations = 300, filename = filename)
  # test simulators prediction accuracy
  print('=='*10)
  print(f'Test the prediction error for the {i}-th simulator')
  print('=='*10)
  test_simulator(filename, args.env, device, args.num_episodes, T = 50)
  simulators.append(filename)

print('=='*10)
print(f'Evaluate Average Reward via MPC')
print('=='*10)
eval_policy(simulators, args.env, args.num_mpc_evals, device)
# Estimate average reward


