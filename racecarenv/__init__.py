from racecar_gym.env import RaceEnv
from .wrapper import get_wrapper
from .rewardshape import get_rewardshaping_wrapper


def get_env(args, conf, **kwargs):
    env = RaceEnv(scenario=args.scenario,
                  reset_when_collision=False if 'collisionStop' in args.scenario else True,
                  **kwargs,)
    env = get_wrapper(env, conf["wrappers"])
    env = get_rewardshaping_wrapper(env, conf["rewardfuncs"])
    return env
                
                


