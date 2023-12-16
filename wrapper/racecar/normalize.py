from .base import BaseWrapper
from typing import Union
import gymnasium as gym
import numpy as np
import cv2

EnvType = Union[gym.Env, BaseWrapper]

class NormalizeWrapper(BaseWrapper):
    """Standardize the observation.
    The observation is standardized to have mean 0 and standard deviation 1.
    Input: (C, H, W)
    Output: (C, H, W) but standardized
    """

    def __init__(self, env: EnvType):
        super().__init__(env)
        C, H, W = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(C, H, W), dtype=np.float32)
        
        assert isinstance(self.observation_space, gym.spaces.Box), f'Only support Box observation space. ' \
                                                                   f'Wrong in {self.env.__class__.__name__}. ' \
                                                                   f'Got {type(self.observation_space)}'


    def standardize_obs(self, obs):
        obs = cv2.normalize(obs, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return obs

    def step(self, action):
        obs, *others = self.env.step(action)
        obs = self.standardize_obs(obs)
        return obs, *others

    def reset(self, *args, **kwargs):
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.standardize_obs(obs)
        return obs, *others
