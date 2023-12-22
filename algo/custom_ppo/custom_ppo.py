import warnings

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, \
    is_image_space_channels_first
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
    is_vecenv_wrapped,
)
import numpy as np
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from gymnasium import spaces


class CustomPPO(PPO):

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose: bool = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                            is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                def is_image_space_channels_first_modified(observation_space: spaces.Box) -> bool:
                    """
                    Check if an image observation space (see ``is_image_space``)
                    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

                    Use a heuristic that channel dimension is the smallest of the three.
                    If second dimension is smallest, raise an exception (no support).

                    :param observation_space:
                    :return: True if observation space is channels-first image, False if channels-last.
                    """
                    smallest_dimension = np.argmin(observation_space.shape).item()
                    if smallest_dimension == 1:
                        warnings.warn(
                            "Treating image space as channels-last, while second dimension was smallest of the three.")
                        return True
                    return smallest_dimension == 0

                wrap_with_vectranspose = is_image_space(
                    env.observation_space) and not is_image_space_channels_first_modified(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env
