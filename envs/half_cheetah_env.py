import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch 

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mass_scale_set=1.0, damping_scale_set= 1.0, max_timesteps = 1000):
        self.t = 0
        self.max_timesteps = max_timesteps
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set  
        self.max_timesteps = max_timesteps

        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set)
        
        self.reset()
        
    def _set_observation_space(self, observation):
        super(HalfCheetahEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation[None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1  * np.square(action).sum()
        reward_run = ob[0]
        reward = reward_run + reward_ctrl

        done = False
        self.t +=1
        if self.t  > self.max_timesteps:
            done = True
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def _set_state(self, state):
        self.prev_qpos[0] = state[1] - state[0] * self.dt
        #state[0] = (state[:1]*self.dt + self.prev_qpos[:1])
        qpos = state[1:10]
        qvel = state[10:]
        self.set_state(qpos, qvel)
        self.t =0 



    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        else:
            return tf.concat([obs[..., 1:2], tf.sin(obs[..., 2:3]), tf.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)

    def obs_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)
        else:
            return tf.concat([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)

    def targ_proc(self, obs, next_obs):
        return np.concatenate([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)

        # random_index = self.np_random.randint(len(self.mass_scale_set))
        # self.mass_scale = self.mass_scale_set[random_index]
        self.mass_scale = self.mass_scale_set

        # random_index = self.np_random.randint(len(self.damping_scale_set))
        # self.damping_scale = self.damping_scale_set[random_index]
        self.damping_scale = self.damping_scale_set
        self.t = 0
        self.change_env()
        return self._get_obs()

    def reward(self, obs, action, next_obs):
        ctrl_cost = 1e-1 * np.sum(np.square(action), axis=-1)
        forward_reward = obs[..., 0]
        reward = forward_reward - ctrl_cost
        return reward
    
    def torch_reward_fn(self):
        def _thunk(obs, act, next_obs):
            ctrl_cost = 1e-1  * torch.sum(torch.square(act), axis=-1)
            forward_reward = next_obs[..., 0]
            reward = forward_reward - ctrl_cost
            return reward
        return _thunk
    
    def torch_done_fn(self):
        def _thunk(next_obs):
            done =  torch.zeros(next_obs.shape[0], device = next_obs.device)
            return done
        return _thunk

    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
    
    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])
    
    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return
