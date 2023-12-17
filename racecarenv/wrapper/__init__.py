from .base import BaseWrapper
from .resize import ResizeWrapper
from .normalize import NormalizeWrapper
from .grayscale import GrayScaleWrapper 
from .framstack import FrameStackWrapper
from .edgedetection import EdgeDetectionWrapper
from .rewardshape import RewardShapingWrapper
from .initmode import InitModeWrapper


from typing import Union
import gymnasium as gym

import inspect


EnvType = Union[gym.Env, BaseWrapper]

def get_wrapper(env: EnvType, wrapper_conf: dict) -> EnvType:    
    # need to consider the order of wrappers
    if 'InitMode' in wrapper_conf:
        env = InitModeWrapper(env, **wrapper_conf['InitMode'])
    if 'Resize' in wrapper_conf:
        env = ResizeWrapper(env, **wrapper_conf['Resize'])
    if 'GrayScale' in wrapper_conf:
        env = GrayScaleWrapper(env)
    if 'EdgeDetection' in wrapper_conf:
        env = EdgeDetectionWrapper(env)
    if 'Normalize' in wrapper_conf:
        env = NormalizeWrapper(env)
    if 'FrameStack' in wrapper_conf:
        env = FrameStackWrapper(env, **wrapper_conf['FrameStack'])
    if 'RewardShaping' in wrapper_conf:
        env = RewardShapingWrapper(env, **wrapper_conf['RewardShaping'])
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
        ResizeWrapper,
        NormalizeWrapper,
        GrayScaleWrapper ,
        FrameStackWrapper,
        EdgeDetectionWrapper,
        RewardShapingWrapper,
        InitModeWrapper
    ]
    
    configs = {}
    for cls in classes:
        class_n = cls.__name__
        class_n = class_n.replace("Wrapper", "")
        configs[class_n] = generate_default_config(cls)
    return configs
