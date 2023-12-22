from .base import BaseRewardFunc


class SpeedRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc, speedbonus: float) -> None:
        super().__init__(reward_func)
        self.speedbonus = speedbonus
        
    
    def speed_reward(self, reward, info, rew_dict):
        if info['velocity'][0] >= self.speedbonus:
            rew_dict['speedbonus'] = 0.00005
            reward += 0.00005
        return reward, rew_dict

    def __call__(self, reward, info):
        reward, rew_dict = self.reward_func(reward, info)
        reward, rew_dict = self.speed_reward(reward, info, rew_dict)
        return reward, rew_dict