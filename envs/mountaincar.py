from tqdm import tqdm
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding 
import math
import torch

class MountainCarEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0, gravity = 0.0025, force = 0.001, max_timesteps = 500):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.t = 0
        self.max_timesteps = max_timesteps
        # self.actions = [1,2,3]
        self.force = force
        self.gravity = gravity

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        self.t += 1 
        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0 * (1-int(done))
        
        if self.t > self.max_timesteps:
            done = True
        
        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self, state=None):
        self.t = 0
        if state is None: self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        else: self.state = state
        return np.array(self.state)

    def _set_state(self, state):
            self.state = state
    
    def torch_reward_fn(self):
        def _thunk(obs, act, next_obs):
            x_threshold = 0.5
            cond1 = next_obs[...,0] >= x_threshold
            cond = cond1.float()
            reward = -1 + cond
            return reward
        return _thunk

    def torch_done_fn(self):
        def _thunk(next_obs):
            done =  torch.zeros(next_obs.shape[0], device = next_obs.device)
            return done
        return _thunk
        
    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55


