from base import BaseWrapper
from typing import Union
import gymnasium as gym
import numpy as np
import cv2

import matplotlib.pyplot as plt

EnvType = Union[gym.Env, BaseWrapper]

class EdgeDetectionWrapper(BaseWrapper):
    """Standardize the observation.
    The observation is standardized to have mean 0 and standard deviation 1.
    Input: (C, H, W)
    Output: (C, H, W) but standardized
    """

    def __init__(self, env: EnvType):
        super().__init__(env)
        C, H, W = self.observation_space.shape
        old_type = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(C, H, W), dtype=old_type)
        
        assert isinstance(self.observation_space, gym.spaces.Box), f'Only support Box observation space. ' \
                                                                   f'Wrong in {self.env.__class__.__name__}. ' \
                                                                   f'Got {type(self.observation_space)}'


    def edgedetect_obs(self, obs):
        # Edge detect        
        obs_ = obs.transpose(1, 2, 0)

        convert = False if isinstance(obs_, np.uint8) else True
        if convert:
            original_dtype = obs_.dtype
            obs_ = obs_.astype(np.uint8)
        
        obs_ = cv2.Canny(obs_, 200, 250)
        obs_ = np.expand_dims(obs_, axis=0)
    
        
        if convert: obs_ = obs_.astype(original_dtype)
        return obs_


    def step(self, action):
        obs, *others = self.env.step(action)
        obs = self.edgedetect_obs(obs)
        return obs, *others

    def reset(self, *args, **kwargs):
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.edgedetect_obs(obs)
        return obs, *others
