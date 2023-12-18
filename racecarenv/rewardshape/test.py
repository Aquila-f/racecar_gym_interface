from .base import BaseRewardFunc


class TestRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc) -> None:
        super().__init__(reward_func)
    
    def test_reward(self, reward, info, rew_dict):
        rew_dict["test_reward"] = reward+1.
        return reward+1., rew_dict

    def __call__(self, reward, info):
        reward, rew_dict = self.reward_func(reward, info)
        reward, rew_dict = self.test_reward(reward, info, rew_dict)
        return reward, rew_dict