from .base import BaseWrapper
import numpy as np
import requests
import json
from typing import Union
import gymnasium as gym

EnvType = Union[gym.Env, BaseWrapper]

class ClientWrapper(BaseWrapper):
    """Standardize the observation.
    The observation is standardized to have mean 0 and standard deviation 1.
    Input: (C, H, W)
    Output: (C, H, W) but standardized
    """

    def __init__(self, env: EnvType, url: str):
        super().__init__(env)
        self.url = url

    def get_obs_from_server(self):
        response = requests.get(f'{self.url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            exit(1)
        response_dict = json.loads(response.text)
        if response_dict.get('terminal'):
            print('Episode finished.')
            exit(1)
        
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)
        return obs


    def send_act_and_get_terminal(self, action):
        # Send an action and receive new observation, reward, and done status
        print(f'Sending action: {action}')
        response = requests.post(f'{self.url}', json={'action': list(action)})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            exit(1)
        result = json.loads(response.text)
        
        terminal = result['terminal']
        return terminal

    def step(self, actions):
        terminal = self.send_act_and_get_terminal(actions)
        if terminal:
            print('Episode finished.')
            exit(1)
        obs = self.get_obs_from_server()
        return obs, 0, False, False, {}

    def reset(self, *args, **kwargs):
        obs = self.get_obs_from_server()
        return obs, {}
