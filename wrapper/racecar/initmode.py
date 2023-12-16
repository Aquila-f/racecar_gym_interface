from .base import BaseWrapper
from typing import Union
import gymnasium as gym


EnvType = Union[gym.Env, BaseWrapper]

class InitModeWrapper(BaseWrapper):
    """A wrapper for training RaceEnv agents with different init modes.
     That is, set the init position.

    Parameters
    ==========
    init_mode: str
        The init mode of the environment. Available options are:
        - 'grid': The car is initialized at the grid.
        - 'random': The car is initialized at a random position.
        - more detail in racecar_gym_competition_env/racecar_gym/bullet/world.py
    """

    def __init__(self, env: EnvType, init_mode: str = 'grid'):
        super().__init__(env)
        self.init_mode = init_mode

    def reset(self, *args, **kwargs):
        # Overwrite the init mode
        if kwargs.get('options'):
            assert kwargs['options'].get('mode') is None, f'Please do not specify mode when calling reset!'
            kwargs['options']['mode'] = self.init_mode
        else:
            kwargs['options'] = {'mode': self.init_mode}
        return self.env.reset(*args, **kwargs)