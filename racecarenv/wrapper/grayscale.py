from .base import BaseWrapper
from typing import Union
import gymnasium as gym
import numpy as np
import cv2

EnvType = Union[gym.Env, BaseWrapper]

class GrayScaleWrapper(BaseWrapper):
    """Gray scale the observation.
    Input: (C, H, W)
    Output: (H, W)
    """

    def __init__(self, env):
        super().__init__(env)
        C, H, W = self.observation_space.shape
        old_type = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, H, W), dtype=old_type)

    @staticmethod
    def obs_grayscale(obs):
        # Resize
        obs_ = obs.transpose(1, 2, 0)

        convert = False if isinstance(obs_, np.uint8) else True
        if convert:
            original_dtype = obs_.dtype
            obs_ = obs_.astype(np.uint8)

        obs_ = cv2.cvtColor(obs_, cv2.COLOR_BGR2GRAY)
        obs_ = np.expand_dims(obs_, axis=0)
        # obs_ = obs_.transpose(2, 0, 1)
        
        if convert: obs_ = obs_.astype(original_dtype)
        return obs_
    
    def step(self, action):
        obs, *others = self.env.step(action)
        obs = self.obs_grayscale(obs)
        return obs, *others

    def reset(self, *args, **kwargs):
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.obs_grayscale(obs)
        return obs, *others