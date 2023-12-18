from typing import Callable


class BaseRewardFunc():
    """ A basic reward function for training RaceEnv agents.
    """

    def __init__(self, reward_func: Callable = None) -> None:
        self.reward_func = reward_func
        self.reward_dict: dict = {}
    
    def reward_function(self, reward, info):
        self.reward_dict["original_reward"] = reward
        return reward

    def __call__(self, reward, info):
        reward = self.reward_function(reward, info)
        return reward, self.reward_dict

    