from .base import BaseRewardFunc


class NoRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc) -> None:
        super().__init__(reward_func)
        
    
    def no_reward(self, reward, info, rew_dict):
        rew_dict["noreward"] = 0.
        return 0., rew_dict

    def __call__(self, reward, info):
        reward, rew_dict = self.reward_func(reward, info)
        reward, rew_dict = self.no_reward(reward, info, rew_dict)
        return reward, rew_dict