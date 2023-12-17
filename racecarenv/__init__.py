from racecar_gym.env import RaceEnv
from .wrapper import get_wrapper


def get_env(scenario, wrapper_conf, **kwargs):
    env = RaceEnv(scenario=scenario,
                  reset_when_collision=False if 'collisionStop' in scenario else True,
                  **kwargs,)
    env = get_wrapper(env, wrapper_conf)
    return env
                
                


