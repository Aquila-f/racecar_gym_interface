from .base import BaseWrapper
from .frameskip import FrameSkipWrapper
from .resize import ResizeWrapper
from .normalize import NormalizeWrapper
from .grayscale import GrayScaleWrapper 
from .framstack import FrameStackWrapper
from .edgedetection import EdgeDetectionWrapper
from .initmode import InitModeWrapper
from .discretizeaction import DiscretizeActionWrapper
from .saveinfo import SaveInfoWrapper
from .client import ClientWrapper

from typing import Union
import gymnasium as gym

import inspect


EnvType = Union[gym.Env, BaseWrapper]

def get_wrapper(env: EnvType, wrapper_config: dict, eval_port: str = None) -> EnvType:    
    # need to consider the order of wrappers
    if eval_port: env = ClientWrapper(env, eval_port)
    for classname, params in wrapper_config.items():
        cls = globals().get(classname, None)
        if cls is None:
            raise ValueError(f'Cannot find class {classname} in wrapper.')
        env = cls(env, **params)

    return env
    
    
def generate_default_config(cls):
    constructor = inspect.signature(cls.__init__)
    config = {}
    for name, param in constructor.parameters.items():
        if name in ['self', 'env']:
            continue
        
        param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
        default_value = param.default if param.default != inspect.Parameter.empty else param_type
        config[name] = default_value
    return config


def return_wrapper_config():
    # register all wrappers here
    classes = [
        InitModeWrapper,
        SaveInfoWrapper,
        FrameSkipWrapper,
        ResizeWrapper,
        GrayScaleWrapper ,
        EdgeDetectionWrapper,
        NormalizeWrapper,
        FrameStackWrapper,
        DiscretizeActionWrapper
    ]
    
    configs = {}
    for cls in classes:
        class_n = cls.__name__
        configs[class_n] = generate_default_config(cls)
    return configs
