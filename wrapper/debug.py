from racecar import *
import matplotlib.pyplot as plt

from racecar_gym.env import RaceEnv
import json



env = RaceEnv(scenario="austria_competition_collisionStop",
                      reset_when_collision=False)



with open("config.json", 'r') as file:
    config = json.load(file)

env = get_env(env, config)

# frame_freq = 25
# number_frame_stack = 5

# env = BaseWrapper(env)
# print("BaseWrapper")
# env = ResizeWrapper(env, 64)
# print("ResizeWrapper")
# env = GrayScaleWrapper(env)
# print("GrayScaleWrapper")
# env = EdgeDetectionWrapper(env)
# print("EdgeDetectionWrapper")
# env = NormalizeWrapper(env)
# print("NormalizeWrapper")
# env = FrameStackWrapper(env, frame_freq, number_frame_stack)
# print("FrameStackWrapper")

obs, info = env.reset()



for _ in range(100):
    # action = env.action_space.sample()
    obs, rewards, dones, truncated, states = env.step((0.01, 1))


# # subplot all of the frames in the obs and show them
for i in range(3+1):
    plt.subplot(1, 3+1, i+1)
    plt.imshow(obs[i])
plt.show()



