from stable_baselines3 import PPO

class CustomPPO(PPO):
    def name(self):
        return 'CustomPPO'

    