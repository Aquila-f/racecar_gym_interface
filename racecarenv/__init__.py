from racecar_gym.env import RaceEnv
from .wrapper import get_wrapper
from .rewardshape import get_rewardshaping_wrapper


def get_env(args, conf, **kwargs):
    if args.savename and 'SaveInfoWrapper' in conf["wrappers"]: conf["wrappers"]["SaveInfoWrapper"]["save_pos_folder_name"] = args.savename
    env = RaceEnv(scenario=args.scenario,
                  reset_when_collision=False if 'collisionStop' in args.scenario else True,
                  **kwargs,)
    env = get_rewardshaping_wrapper(env, conf["rewardfuncs"])
    env = get_wrapper(env, conf["wrappers"], args.eport)
    return env
                
                


