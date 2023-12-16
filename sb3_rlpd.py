import json
import importlib

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def load_algorithm_class(class_name):
    module = importlib.import_module("algo")
    return getattr(module, class_name)

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Dynamically load the algorithm class
    algo_class = load_algorithm_class(config['algo'])

    vec_eval_env = SubprocVecEnv([lambda: get_env(args) for _ in range(5)])
    vec_train_env = SubprocVecEnv([lambda: get_env(args) for _ in range(args.n_train)])

    

    # Create an instance of the algorithm with the specified parameters
    algorithm_instance = algo_class(**config['parameters'])
    print(algorithm_instance.name())

    # ... rest of your main logic ...

if __name__ == "__main__":
    main("config.json")
