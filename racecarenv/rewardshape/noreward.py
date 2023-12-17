from .base import BaseRewardFunc


class NoRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc) -> None:
        super().__init__(reward_func)
        
    
    def no_reward(self, reward, info):
        return 0.

    def __call__(self, reward, info):
        reward = self.reward_func(reward, info)
        reward = self.no_reward(reward, info)
        return reward