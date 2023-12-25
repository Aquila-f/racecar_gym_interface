from .base import BaseWrapper
from typing import Union
import gymnasium as gym

EnvType = Union[gym.Env, BaseWrapper]


class FrameSkipWrapper(BaseWrapper):
    """Skip frames."""

    def __init__(self, env: EnvType, frame_skip: int):
        super().__init__(env)
        self.frame_skip = frame_skip
        assert self.frame_skip > 0, f'frame_repeat must be greater than 0. Got {self.frame_skip}'

    def step(self, action):
        obs, rew, terminal, trunc, info = None, 0., None, None, None
        for _ in range(self.frame_skip):
            obs, rew_, terminal, trunc, info = self.env.step(action)
            rew += rew_
            if terminal:
                break
        return obs, rew, terminal, trunc, info
