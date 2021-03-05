from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, categorical
import numpy as np
from mpc.utils import assert_shape

gamma = 1.0


class DynamicsFunc(ABC):
    """The user should implement this to specify the dynamics of the system, and the objective function."""

    @abstractmethod
    def step(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the next state and cost of the action.

        :param states: [N x state dimen] the current state, where N is the batch dimension
        :param actions: [N x action dimen] the action to perform, where N is the batch dimension
        :returns:
            next states [N x state dimen],
            costs [N x 0]
        """
        pass


Rollouts = namedtuple('Rollouts', 'trajectories actions objective_costs')


class RolloutFunction:
    """Computes rollouts (trajectory, action sequence, cost) given an initial state and parameters.

    This logic is in a separate class to allow multithreaded rollouts, though this is not currently implemented.
    """

    def __init__(self, dynamics: list, state_dimen: int, action_dimen: int,
                 time_horizon: int, num_rollouts: int, num_actions = None,max_action = None, mountain_car = False):
        self._dynamics = dynamics
        self.no_models = None# len(dynamics)
        self._state_dimen = state_dimen
        self._action_dimen = action_dimen
        self._time_horizon = time_horizon
        self._num_rollouts = num_rollouts
        self.num_actions = num_actions
        self.max_action = max_action
        self.mountain_car = mountain_car
    def perform_rollouts(self, args: Tuple[Tensor, Tensor, Tensor]) -> Rollouts:
        """Samples a trajectory, and returns the trajectory and the cost.

        :param args: (initial_state [state_dimen], action means, action stds)
        :returns: (sequence of states, sequence of actions, cost)
        """
        initial_state, means, stds = args
        initial_states = initial_state.repeat((self._num_rollouts, 1))
        trajectories, actions, objective_costs, next_costs= self._sample_trajectory(initial_states, means, stds)
        
        return Rollouts(trajectories, actions, objective_costs), next_costs

    def perform_rollouts_disceret(self, initial_state: Tensor, previous_action ) -> Rollouts:
        """Samples a trajectory, and returns the trajectory and the cost.

        :param args: (initial_state [state_dimen], action means, action stds)
        :returns: (sequence of states, sequence of actions, cost)
        """
      
        initial_states = initial_state.repeat((self._num_rollouts, 1))
        trajectories, actions, objective_costs = self._sample_trajectory_disceret(initial_states, previous_action)
        # temp fix
        
        return Rollouts(trajectories, actions, objective_costs)

    def _sample_trajectory(self, initial_states: Tensor, means: Tensor, stds: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Randomly samples T actions and computes the trajectory.

        :returns: (sequence of states, sequence of actions, costs)
        """
        # assert_shape(initial_states, (self._num_rollouts, self._state_dimen))
        # assert_shape(means, (self._time_horizon, self._action_dimen))
        # assert_shape(stds, (self._time_horizon, self._action_dimen))

        actions = Normal(means, stds).sample(sample_shape=(self._num_rollouts,))
        # assert_shape(actions, (self._num_rollouts, self._time_horizon, self._action_dimen))
        # print(torch.abs(torch.max(actions,1)[0]))
        if self.max_action is not None:
            actions = actions.clip(-self.max_action, self.max_action)
            #print(actions.max()) 
        # One more state than the time horizon because of the initial state.
        trajectories = torch.empty((self._num_rollouts, self._time_horizon + 1, self._state_dimen),
                                   device=initial_states.device)
        trajectories[:,  0, :] = initial_states
        objective_costs = torch.zeros(( self._time_horizon,self._num_rollouts,), device=initial_states.device)
        dones = torch.zeros(( self._num_rollouts,), device=initial_states.device)
        # print(actions.shape)
        for t in range(self._time_horizon):
            #for d,dynamic in enumerate(self._dynamics):
                next_states, costs, done = self._dynamics.step(trajectories[:, t, :], actions[:, t, :])
                # assert_shape(next_states, (self._num_rollouts, self._state_dimen))
                # assert_shape(costs, (self._num_rollouts,))
                trajectories[ :,t + 1, :] = next_states
                dones[:] =  torch.maximum(done,dones[:])
                objective_costs[t,:] = (gamma)**t*costs*(1-dones[:]) #+ dones[d,:]*100 
        # objective_costs = torch.max(objective_costs,0)
        # print(trajectories[:,:,:,:].shape)
        #num_comparisons =  int(self.no_models * (self.no_models+1)/2)
        """
        distance = torch.zeros((num_comparisons, self._num_rollouts,), device=initial_states.device)
        index = -1
        for i in range(self.no_models):
            for j in range(i+1, self.no_models):
                index +=1
                #distance[index,:]  = torch.square(trajectories[i,:,:,0] - trajectories[j,:,:,0]).mean(1)
                distance[index,:] = torch.sqrt(torch.square(objective_costs[i,:] - objective_costs[j,:]))
        distance =  torch.max(distance,0)[0]
        """
        #costs_var = torch.std(trajectories[:,:,10,0], 0)
        #costs_mean = torch.median(trajectories[:,:,10,0], 0)[0]
        #distance = costs_var/torch.abs(costs_mean)
        #print(costs_var.shape, costs_mean.shape, distance.shape, distance)
        #objective_costs = torch.mean(objective_costs,0)
        next_costs= objective_costs[0,:].clone()
        objective_costs = torch.sum(objective_costs,0)
        #objective_costs = torch.max(objective_costs,0)[0]
        #print('distance', distance.shape, distance)
        #print('min',objective_costs.shape,  objective_costs.min(), objective_costs.max(), max(torch.median(distance).item(),10))    
        #objective_costs[distance> 0.1] = 1000  #max(torch.median(distance).item(),0.5)] = 1000 
        #print('min',objective_costs.shape,  objective_costs.min(), objective_costs.max())

        return trajectories[:,:,:], actions, objective_costs#, next_costs

    def _sample_trajectory_disceret(self, initial_states: Tensor, previous_action ) -> Tuple[Tensor, Tensor, Tensor]:
            """Randomly samples T actions and computes the trajectory.

            :returns: (sequence of states, sequence of actions, costs)
            """
            # assert_shape(initial_states, (self._num_rollouts, self._state_dimen))
            actions = categorical.Categorical(torch.ones(self.num_actions)/self.num_actions).sample(sample_shape=(self._num_rollouts,self._time_horizon,1))
            if previous_action is not None:
                actions[0,:-1,0] = previous_action[1:self._time_horizon,0]
            # assert_shape(actions, (self._num_rollouts, self._time_horizon, self._action_dimen))

            # One more state than the time horizon because of the initial state.
            trajectories = torch.empty((  self._num_rollouts, self._time_horizon + 1, self._state_dimen),
                                       device=initial_states.device)
            trajectories[:, 0, :] = initial_states
            objective_costs = torch.zeros((self._time_horizon, self._num_rollouts,), device=initial_states.device)
            dones = torch.zeros((self._num_rollouts,), device=initial_states.device)

            for t in range(self._time_horizon):
                #for d,dynamic in enumerate(self._dynamics):

                    next_states, costs, done = self._dynamics.step(trajectories[:, t, :], actions[:, t, 0])

                    # assert_shape(next_states, (self._num_rollouts, self._state_dimen))
                    # assert_shape(costs, (self._num_rollouts,))
                    
                    trajectories[ :, t + 1, :] = next_states
                    dones[:] =  torch.maximum(done,dones[:])
                    #print(dones[d,:], d, t)
                    # TODO: worry about underflow.
                    objective_costs[t,:] = (gamma)**t*costs*(1-dones[:])
    
            
            if self.mountain_car:
                objective_costs = objective_costs - 0.01*torch.max(trajectories[:,:,0],1)[0]
            
            # objective_costs = torch.mean(objective_costs,1)#[0]
            objective_costs = torch.sum(objective_costs,0)
            
            # objective_costs = torch.max(objective_costs,0)[0]
            # print('min', objective_costs.min(), objective_costs.max())
            # print('min', dones[:,:].sum(0), dones.min())
            return trajectories, actions, objective_costs

    

class MPC_M:
    """An MPC implementation which supports polytopic constraints for trajectories, using cross-entropy optimisation.

    This method is based on 'Constrained Cross-Entropy Method for Safe Reinforcement Learning'; Wen, Topcu. It includes
    the constraints as additional optimisation objectives, which must be satisfied before the trajectory cost is
    optimised.
    """

    def __init__(self, dynamics_func: list, state_dimen: int, action_dimen: int,
                 time_horizon: int, num_rollouts: int, num_elites: int, num_iterations: int, disceret_action: bool =  False,
                 rollout_function: RolloutFunction = None, max_action = None, mountain_car = False):
        """Creates a new instance.

        :param state_dimen: number of dimensions of the state
        :param action_dimen: number of dimensions of the actions
        :param time_horizon: T, number of time steps into the future that the algorithm plans
        :param num_rollouts: number of trajectories that the algorithm samples each optimisation iteration
        :param num_elites: number of trajectories, i.e. the best m, which the algorithm refits the distribution to
        :param num_iterations: number of iterations of CEM
        :param rollout_function: only set in unit tests
        """
        self._action_dimen = action_dimen
        self._time_horizon = time_horizon
        self._num_rollouts = num_rollouts
        self._num_elites = num_elites
        self._num_iterations = num_iterations
        self.disceret_action = disceret_action
        self._no_actions = None
        if self.disceret_action:
            self._no_actions = self._action_dimen
            self._action_dimen  = 1

        if rollout_function is None:
            rollout_function = RolloutFunction(dynamics_func, state_dimen, self._action_dimen, time_horizon,
                                               num_rollouts, self._no_actions, max_action = max_action, mountain_car = mountain_car)
        self._rollout_function = rollout_function
        self.mean = None
        self.previous_action = None
        self.max_action = max_action
    
    def optimize_trajectories(self, initial_state: Tensor) -> List[Rollouts]:
        """
        The trajectories this function returns are not guaranteed to be safe. Thus, normally, do not call this method
        directly. Instead, call get_actions().

        :returns: A list of rollouts from each CEM iteration. The final step is last.
        """
        
        if self.disceret_action:
            #######################
            #perform Shooting method#
            #######################
            rollouts = self._rollout_function.perform_rollouts_disceret(initial_state, self.previous_action)
            return rollouts
                
        else:
            #######################
            #perform CEM#
            #######################
            # if self.mean is None:
            self.mean = torch.zeros((self._time_horizon, self._action_dimen), device=initial_state.device)
            # print('init mean')
            
            means = self.mean
            init_stds = (1/3)*torch.ones((self._time_horizon, self._action_dimen), device=initial_state.device)
            stds = init_stds.clone()
            rollouts_by_iteration = []
            lower_bound = -self.max_action
            upper_bound = self.max_action
            for i in range(self._num_iterations):
                # var = torch.square(stds)
                #lb_dist, ub_dist = means - lower_bound, upper_bound - means
                #stds = torch.minimum(torch.minimum(lb_dist/2,ub_dist/2), stds) 
                # print('stds', stds[0,:])
                rollouts = self._rollout_function.perform_rollouts((initial_state, means, stds))
                elite_rollouts = self._select_elites(rollouts)
                means = elite_rollouts.actions.mean(dim=0)
                # print('means', means[0,:])
                #stds = elite_rollouts.actions.std(dim=0)
                # print(stds.max(), stds.min())
                # rollouts_by_iteration.append(rollouts)
                # stds = (1/(i+3)) *torch.ones((self._time_horizon, self._action_dimen), device=initial_state.device)
                    
            return elite_rollouts #, next_costs

    def _select_elites(self, rollouts: Rollouts) -> Rollouts:
        """Returns the elite rollouts.

        """
        _, sorted_ids_of_feasible_ids = rollouts.objective_costs[:].squeeze().sort()
        elites_ids = sorted_ids_of_feasible_ids[0:self._num_elites].squeeze()
        #print(elites_ids, rollouts.objective_costs[elites_ids], rollouts.objective_costs.min())
        return Rollouts(rollouts.trajectories[elites_ids], rollouts.actions[elites_ids],
                        rollouts.objective_costs[elites_ids])
   
    def get_actions(self, state: Tensor) -> Tuple[Union[Tensor, None], List[Rollouts]]:
        """Computes the approximately optimal actions to take from the given state.

        The sequence of actions is guaranteed to be safe wrt to the constraints.

        :param state: [state dimen], the initial state to plan from
        :returns: (the actions [N x action dimen] or None if we didn't find a safe sequence of actions,
                   rollouts by iteration as returned by optimize_trajectories())
        """
        rollouts_by_iteration = self.optimize_trajectories(state)

        # Use the rollouts from the final optimisation step.
        # min_rollout = [rollouts_.objective_costs.min() for rollouts_ in rollouts_by_iteration]
        # print('min', min_rollout)
        # rollouts = rollouts_by_iteration[np.argmin(min_rollout)]
        rollouts = rollouts_by_iteration
        
        
        costs, sorted_ids_of_feasible_ids = rollouts.objective_costs[:].sort()
        #print(rollouts.objective_costs.shape)
        best_rollout_id = sorted_ids_of_feasible_ids[0].item()
        actions = rollouts.actions
        action = actions[best_rollout_id]
        #print('next cost ', next_costs[best_rollout_id])
        if not self.disceret_action:  
            #print(self.mean.shape, actions[best_rollout_id].shape)
            self.mean[:-1,:] = actions[best_rollout_id][1:,:]
            self.mean[:,:] = 0
            #print('best action',rollouts.trajectories[best_rollout_id,:,0] )
            action = actions.mean(0)
            # print(action.shape)
        else:
            self.previous_action = actions[best_rollout_id]
            #print('best action',rollouts.trajectories[0,best_rollout_id,1,:] )
        
        return action, rollouts.objective_costs[best_rollout_id]
    
