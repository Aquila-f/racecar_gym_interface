from .base import BaseRewardFunc


class DistanceRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc, mindistance: float) -> None:
        super().__init__(reward_func)
        self.mindistance = mindistance
        
    
    def distance_reward(self, reward, info, rew_dict):
        if info['minlidar'] >= self.mindistance:
            rew_dict['distance'] = 0.0001
            reward += 0.0001
        return reward, rew_dict

    def __call__(self, reward, info):
        reward, rew_dict = self.reward_func(reward, info)
        reward, rew_dict = self.distance_reward(reward, info, rew_dict)
        return reward, rew_dict