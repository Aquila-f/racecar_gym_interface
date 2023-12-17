from .base import BaseRewardFunc
from .test import TestRewardFunc
from .noreward import NoRewardFunc
from .rewardshapeWrapper import RewardShapingWrapper

import inspect
from typing import Union

from ..wrapper import BaseWrapper
import gymnasium as gym


EnvType = Union[gym.Env, BaseWrapper]

def get_rewardshaping_func(env: EnvType, rewardfunc_config: dict) -> EnvType:
    # need to consider the order of rewardfuncs
    reward_func = BaseRewardFunc()

    for classname, params in rewardfunc_config.items():
        cls = globals().get(classname, None)
        if cls is None:
            raise ValueError(f'Cannot find class {classname} in rewardshape.')
        reward_func = cls(reward_func, **params)

    
    env = RewardShapingWrapper(env, reward_func)
    return env




def generate_default_config(cls):
    constructor = inspect.signature(cls.__init__)
    config = {}
    for name, param in constructor.parameters.items():
        if name in ['self', 'reward_func']:
            continue
        
        param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
        default_value = param.default if param.default != inspect.Parameter.empty else param_type
        config[name] = default_value
    return config


def return_rewardfunc_config():
    # register all reward shaping function here
    classes = [
        TestRewardFunc,
        NoRewardFunc
    ]
    
    configs = {}
    for cls in classes:
        class_n = cls.__name__
        # class_n = class_n.replace("", "")
        configs[class_n] = generate_default_config(cls)
    return configs
