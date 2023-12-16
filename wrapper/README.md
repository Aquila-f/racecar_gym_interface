# Wrapper with Inheritance - README

This README provides an overview of the Wrapper with Inheritance project.

## Introduction

The Wrapper with Inheritance project aims to demonstrate the concept of inheritance in software development. In this project, we will create a wrapper class that inherits from a base wrapper class.
You can customlize your own wrapper by inherit the basewrapper

## Racecar

we provide some useful wrapper for using the wrapper


### worning

the sequence of using multiple wrappers is improtant
if with wrong sequence of wrapper, it will cause error.
please using debug to try the different wrapper when using multiple wrapper


### create your own customWrapper

class CustomWrapper(BaseWrapper):
    """ custom wrapper

    """

    def __init__(self, env: EnvType):
        super().__init__(env)
        # you may need to get the original observation and reset it
        C, H, W = self.observation_space.shape
        self.observation_space = # [TODO]
        



    def custom_obs_function(self, obs):
        # [TODO]

    def step(self, action):
        obs, *others = self.env.step(action)
        obs = self.custom_obs_function(obs)
        return obs, *others

    def reset(self, *args, **kwargs):
        obs, *others = self.env.reset(*args, **kwargs)
        obs = self.custom_obs_function(obs)
        return obs, *others