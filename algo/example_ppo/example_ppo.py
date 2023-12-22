from stable_baselines3 import PPO

class ExamplePPO(PPO):
    def name(self):
        return 'CustomPPO'

    