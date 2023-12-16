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
import json

classes = [
    ResizeWrapper,
    NormalizeWrapper,
    GrayScaleWrapper ,
    FrameStackWrapper,
    EdgeDetectionWrapper,
    RewardShapingWrapper,
    InitModeWrapper
]


EnvType = Union[gym.Env, BaseWrapper]

def get_env(env: EnvType, wrapper_conf: dict) -> EnvType:
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
    # Get the constructor of the class
    constructor = inspect.signature(cls.__init__)
    
    # Create a default config dictionary, skipping 'self' and 'env'
    config = {}
    for name, param in constructor.parameters.items():
        if name in ['self', 'env']:
            continue

        # Determine the parameter type as a string
        param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'

        # Use the default value if provided, else use the parameter type as placeholder
        default_value = param.default if param.default != inspect.Parameter.empty else param_type
        config[name] = default_value
    
    
    return config


def return_wrapper_config(classes: list):
    configs = {}
    for cls in classes:
        n = cls.__name__
        n = n.replace("Wrapper", "")
        configs[n] = generate_default_config(cls)

    return configs
