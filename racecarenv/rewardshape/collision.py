from .base import BaseRewardFunc


class CollisionRewardFunc(BaseRewardFunc):
    """A reward function for testing RaceEnv agents.
    """
    
    def __init__(self, reward_func: BaseRewardFunc, base_penalty: float) -> None:
        super().__init__(reward_func)
        self.base_penalty = -base_penalty
    
    def collision_reward(self, reward, info, rew_dict):
        tmpreward = 0.
        if info["wall_collision"]:
            pv = info["prev_velocity"][0]
            if pv > 0.: tmpreward = (1-(1./(pv+1.)))*self.base_penalty

        rew_dict["collision"] = tmpreward
        reward += tmpreward
        return reward, rew_dict

    def __call__(self, reward, info):
        reward, rew_dict = self.reward_func(reward, info)
        reward, rew_dict = self.collision_reward(reward, info, rew_dict)
        return reward, rew_dict