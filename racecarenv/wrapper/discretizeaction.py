from .base import BaseWrapper
from typing import Union, List, Tuple
import gymnasium as gym
import os
import warnings
from dataclasses import dataclass

EnvType = Union[gym.Env, BaseWrapper]

@dataclass
class ActionDef:
    """Define an action"""
    action: str
    motor: Tuple[float, float]
    steering: Tuple[float, float]
    key: str

    def __repr__(self):
        return f'{self.action}: {self.key} ({self.motor}, {self.steering})'
    



class DiscretizeActionWrapper(BaseWrapper):
    """Discretize the action space.
    Input: a list of ActionDef.
    The order of the action is the same as the order of the action definitions.
    Note that the action space is overwritten by the action definitions even the action space has been modified.
    """
    DISCRETE_ACTION_DEF_DIR = 'config/discrete_action'

    @staticmethod
    def get_motor_steering_by_discrete_action(action_definitions: List[ActionDef], action: int) -> Tuple[float, float]:
        """Get the motor and steering from the action definitions."""
        act_to_motor_steering = {act_def.action: (act_def.motor, act_def.steering) for act_def in action_definitions}
        return act_to_motor_steering[action]

    @staticmethod
    def load_action_definition(name: str) -> List[ActionDef]:
        """Load the action definition from a file.
        The file should be a yaml file. (Refer to ``config/discrete_action``)
        """
        assert '_' not in os.path.basename(name), f'Please do not use "_" in the file name.'
        import yaml
        with open(os.path.join(DiscretizeActionWrapper.DISCRETE_ACTION_DEF_DIR, f'{name}.yaml'), 'r') as f:
            action_config = yaml.load(f, Loader=yaml.FullLoader)
        print(action_config)
        action_definitions = []
        for act, act_info in action_config.items():
            action_def = ActionDef(act, act_info['motor'], act_info['steering'], act_info['key'])
            action_definitions.append(action_def)
        return action_definitions

    def __init__(self, env: EnvType, action_conf_name: str = "# config discrete action"):
        super().__init__(env)
        if isinstance(action_conf_name, str):  # Load from a file
            action_definitions = self.load_action_definition(action_conf_name)
        self.action_definitions: List[ActionDef] = action_definitions
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            warnings.warn(f'The action space is already discrete. '
                          f'It will be overwritten by the action definitions.')
        self.action_space = gym.spaces.Discrete(len(self.action_definitions))

    def print_action_def(self):
        for action_def in self.action_definitions:
            print(action_def)

    def step(self, action):
        motor, steering = self.action_definitions[action].motor, self.action_definitions[action].steering
        obs, *others = self.env.step((motor, steering))
        return obs, *others