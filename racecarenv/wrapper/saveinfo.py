from .base import BaseWrapper
from typing import Union
import gymnasium as gym
import os
from datetime import datetime
import numpy as np

EnvType = Union[gym.Env, BaseWrapper]


class SaveInfoWrapper(BaseWrapper):
    def __init__(self, env: EnvType, save_pos_folder_name: str):
        super().__init__(env)
        self.save_pos_name = save_pos_folder_name
        self.save_dir = os.path.join('record', save_pos_folder_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.pos_history = []

    @staticmethod
    def save_pose_only_info(pos_history, save_dir):
        time_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
        rand_str = f'{np.random.randint(0, 1000):03d}'
        save_path = os.path.join(save_dir, f'{time_str}_{rand_str}')
        pos_history = np.array(pos_history)
        np.save(save_path, pos_history)

    def step(self, action):
        obs, rew, terminal, trunc, info = self.env.step(action)
        self.pos_history.append(info['pose'])
        if terminal and len(self.pos_history) > 0:
            self.save_pose_only_info(self.pos_history, self.save_dir)
            self.pos_history = []
        return obs, rew, terminal, trunc, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.pos_history = [info['pose']]
        return obs, info