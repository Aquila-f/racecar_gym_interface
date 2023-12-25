from .base import BaseRewardFunc


class DistanceRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc, mindistance: float) -> None:
        super().__init__(reward_func)
        self.mindistance = mindistance
        
    
    def distance_reward(self, reward, info, rew_dict):
        tmpreward = 0.
        if info['minlidar'] >= self.mindistance: tmpreward = 0.000015
        rew_dict['distance'] = tmpreward
        reward += tmpreward
        return reward, rew_dict

    def __call__(self, reward, info):
        reward, rew_dict = self.reward_func(reward, info)
        reward, rew_dict = self.distance_reward(reward, info, rew_dict)
        return reward, rew_dict