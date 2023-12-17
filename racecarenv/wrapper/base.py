from typing import Union
import gymnasium as gym
from racecar_gym.env import RaceEnv

EnvType = Union[RaceEnv, gym.Env]

class BaseWrapper(gym.Env):
    """A basic wrapper for training RaceEnv agents.

    Parameters
    ==========
    env: Type[gym.Env]
        The environment to be wrapped. (RaceEnv or Wrapper)
    """

    def __init__(self, env: EnvType):
        self.env: EnvType = env

    def __getattr__(self, item):
        """Forward the attribute to the wrapped environment."""
        return getattr(self.env, item)

    def step(self, action):
        return self.env.step(action)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.render()
    