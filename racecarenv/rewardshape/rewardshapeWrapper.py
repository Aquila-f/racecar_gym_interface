from ..wrapper.base import BaseWrapper
from typing import Callable, Dict, Union
import gymnasium as gym
EnvType = Union[gym.Env, BaseWrapper]

class RewardShapingWrapper(BaseWrapper):
    """Shape the reward.
    Input: a reward shaping function. The function should have the following signature:
        def reward_shaping_func(reward: float, info: Dict) -> float
    """

    def __init__(self, env: EnvType, reward_shaping_func: Callable[[Union[float], Dict], float]):
        super().__init__(env)
        self.reward_shaping_func = reward_shaping_func

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        reward, _ = self.reward_shaping_func(float(reward), info)
        return obs, reward, done, trunc, info