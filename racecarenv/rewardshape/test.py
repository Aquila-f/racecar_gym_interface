from .base import BaseRewardFunc


class TestRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc) -> None:
        super().__init__(reward_func)
        
    
    def test_reward(self, reward, info):
        self.reward_dict["test_reward"] = reward+1.
        return reward+1.

    def __call__(self, reward, info):
        reward, _ = self.reward_func(reward, info)
        reward = self.test_reward(reward, info)
        return reward, self.reward_dict