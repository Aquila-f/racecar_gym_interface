from rewardshape import TestRewardFunc, BaseRewardFunc




reward_f = BaseRewardFunc()
reward_f = TestRewardFunc()
# reward_f = BaseRewardFunc()
# reward_f = TestRewardFunc(reward_f)

print(reward_f(1,2))