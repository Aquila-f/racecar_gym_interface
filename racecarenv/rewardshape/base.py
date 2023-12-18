from typing import Callable


class BaseRewardFunc():
    """ A basic reward function for training RaceEnv agents.
    """

    def __init__(self, rf: Callable = None) -> None:
        self.reward_func = rf
    
    def reward_function(self, reward, info):
        return reward

    def __call__(self, reward, info):
        reward = self.reward_function(reward, info)
        reward_dict = {}
        reward_dict["original_reward"] = reward
        return reward, reward_dict

    