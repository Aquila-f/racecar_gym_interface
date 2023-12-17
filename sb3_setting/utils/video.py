import os
from datetime import datetime
from typing import Tuple, Optional
import gymnasium as gym
import numpy as np

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError as e:
    import gymnasium as gym

    raise gym.error.DependencyNotInstalled(
        "moviepy is not installed, run `pip install moviepy`"
    ) from e


def record_evaluate_policy(policy,
                           env: gym.Env,
                           video_dir: str,
                           n_eval_episodes=1,
                           deterministic=True,
                           name_suffix: Optional[str] = None) -> Tuple[float, float]:
    """
    Record a video of an agent's performance and returns the mean reward.

    :param policy: The sb3 policy
    :param env: (gym.Env) The gym environment
    :param video_dir: (str) Where to save the video
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :return: (float) Mean reward and std error
    """
    start_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")

    episode_rewards = []

    # Evaluate the policy and record the rendered images to make a video
    for i_eval in range(n_eval_episodes):
        obs, info = env.reset()
        terminal = False
        rendered_frames = []
        rewards = []
        while not terminal:
            action, _ = policy.predict(obs.copy())
            obs, rew, terminal, trunc, info = env.step(action)
            rewards.append(rew)
            rendered_frames.append(env.render())

        total_reward = sum(rewards)

        filename = f'{start_time}_{i_eval}_{total_reward:.4f}'
        if isinstance(name_suffix, str):
            filename += f'_{name_suffix}'
        filename += '.mp4'

        clip = ImageSequenceClip(rendered_frames, fps=30)
        clip.write_videofile(os.path.join(video_dir, filename), logger="bar")

        episode_rewards.append(total_reward)

    mean_reward = sum(episode_rewards) / n_eval_episodes
    std_reward = np.std(episode_rewards) / np.sqrt(n_eval_episodes)

    return mean_reward, std_reward
