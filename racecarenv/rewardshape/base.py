from typing import Callable


class BaseRewardFunc():
    """ A basic reward function for training RaceEnv agents.
    """

    def __init__(self, reward_func: Callable = None) -> None:
        self.reward_func = reward_func

    def __call__(self, reward, info):
        return reward

    