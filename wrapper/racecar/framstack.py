from base import BaseWrapper
from typing import Union
import gymnasium as gym
import numpy as np
from collections import deque
from copy import deepcopy

EnvType = Union[gym.Env, BaseWrapper]

class FrameStackWrapper(BaseWrapper):
    """Stack the frames.

    Parameters
    ==========
    frame_freq: int
        The frequency of stacking frames.
    n_frame_stack: int
        The number of previous frames to be stacked.
        Total number of frames = n_frame_stack + 1

    Example
    =======
    For example, if frame_freq = 4 and n_frame_stack = 3, (assume the step = t now)
    then the frame stack will contain the following frames:
    - Obs at step t
    - Obs at step t-4
    - Obs at step t-8
    - Obs at step t-12
    """

    def __init__(self,
                 env: EnvType,
                 frame_freq: int,
                 n_frame_stack: int):
        super().__init__(env)
        self.frame_freq = frame_freq
        self.n_frame_stack = n_frame_stack
        # Assuming the observation space is defined in env
        C, H, W = self.observation_space.shape
        self.obs_hist_size = (self.n_frame_stack * frame_freq + 1)*C
        old_type = self.observation_space.dtype
        
        self.obs_shape = (C, H, W)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((self.n_frame_stack+1)*C, H, W), dtype=old_type)
        
        self.obs_history = deque(maxlen=self.obs_hist_size)

    def init_frame_stack(self):
        self.obs_history.clear()

    def append_frame_stack(self, obs):
        # Append a new observation to the history, automatically removing the oldest
        self.obs_history.append(obs)

    def get_frame_stack_obs(self, new_obs):
        self.append_frame_stack(new_obs)

        if len(self.obs_history) < self.obs_hist_size:
            while len(self.obs_history) < self.obs_hist_size:
                self.append_frame_stack(new_obs)
        # Stack along a new dimension to keep the frames separate

        stacked_obs = np.concatenate(list(self.obs_history)[::self.frame_freq], axis=0)        
        return stacked_obs

    def step(self, action):
        obs, *others = self.env.step(action)
        stack_obs = self.get_frame_stack_obs(obs)
        return stack_obs, *others

    def reset(self, *args, **kwargs):
        obs, *others = self.env.reset(*args, **kwargs)
        self.init_frame_stack()
        stack_obs = self.get_frame_stack_obs(obs)
        return stack_obs, *others