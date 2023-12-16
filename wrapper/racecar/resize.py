from .base import BaseWrapper
from typing import Union, Tuple
import gymnasium as gym
import cv2
import numpy as np

EnvType = Union[gym.Env, BaseWrapper]


class ResizeWrapper(BaseWrapper):
    """Resize the observation.
    Input: (C, H, W)
    Output: (C, H', W')
    """

    def __init__(self, env: EnvType, resize_shape: Union[Tuple[int, int], int]):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Box), f'Only support Box observation space. ' \
                                                                   f'Wrong in {self.env.__class__.__name__}. ' \
                                                                   f'Got {type(self.observation_space)}'
        
        self.resize_shape = resize_shape if isinstance(resize_shape, tuple) else (resize_shape, resize_shape)
        C, H, W = self.observation_space.shape
        old_type = self.observation_space.dtype
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(C, *self.resize_shape), dtype=old_type)

        

    @staticmethod
    def obs_resize(obs, resize_shape):
        # Resize
        obs_ = obs.transpose(1, 2, 0)

        convert = False if isinstance(obs_, np.uint8) else True
        if convert:
            original_dtype = obs_.dtype
            obs_ = obs_.astype(np.uint8)

        obs_ = cv2.resize(obs_, resize_shape, interpolation=cv2.INTER_AREA)
        obs_ = obs_.transpose(2, 0, 1)
        
        if convert: obs_ = obs_.astype(original_dtype)
        return obs_

    def step(self, action):
        obs, *others = self.env.step(action)
        obs = self.obs_resize(obs, self.resize_shape)
        return obs, *others

    def reset(self, *args, **kwargs):
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.obs_resize(obs, self.resize_shape)
        return obs, *others
    