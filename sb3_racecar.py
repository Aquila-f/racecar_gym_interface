import json
import importlib

import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from algo import return_algo_config
from racecarenv import get_env
from algo import get_algo
from racecarenv.rewardshape import get_rewardshaping_func
from racecarenv.wrapper import return_wrapper_config
from racecarenv.rewardshape import return_rewardfunc_config

from datetime import datetime
from pathlib import Path
import os

from sb3_setting.utils import CustomEvalRewardShapingCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import matplotlib.pyplot as plt

def evaluation_client(args):
    if not args.eport: raise ValueError('Please specify the server ip address.')
    if not args.epath: raise ValueError('Please specify the eval model path.')

    with open(args.conf, 'r') as file:
        config = json.load(file)
    
    args.scenario = "austria_competition"
    env = get_env(args, config)

    model = get_algo(config['algo'])
    model = model.load(args.epath)


    # evaluation info
    print("===== Evaluation Info =====")
    print(f"Scenario: {args.scenario}")
    print(f"Server IP: {args.eport}")
    print(f"Model Path: {args.epath}")
    print(f"Evaluation Config: {args.conf}")
    print("===========================\n\n")

    # count down 3 seconds
    import time
    for i in range(3):
        print(f"Start evaluation in {3-i} seconds...")
        time.sleep(1)

    print("===== Start Evaluation =====")
    from stable_baselines3.common.evaluation import evaluate_policy
    _, _ = evaluate_policy(model, env, n_eval_episodes=1)
    print("===== Evaluation Done =====")
    exit(0)


def debug(args: argparse.Namespace):
    # Load configuration
    with open(args.conf, 'r') as file:
        config = json.load(file)

    env = get_env(args, config)
    obs, info = env.reset()
    print(obs.shape)

    while True:
        action = env.action_space.sample()
        
        obs, rewards, dones, truncated, states = env.step(action)

        # if 'minlidar' in states: print(states['minlidar'], states['obstacle'])
        # else: print(states["obstacle"])
        
        if 'prev_velocity' in states: print("[", states['prev_velocity'][0], states['velocity'][0], "]")
        else: print(states['velocity'][0])
        
        # print(states)
        
        if dones:
            obs, info = env.reset()
            print(rewards)
            a = input()
            obs_ = obs.transpose(1, 2, 0)
            plt.imshow(obs_)
            plt.show()

    # # subplot all of the frames in the obs and show them
    for i in range(3+1):
        plt.subplot(1, 3+1, i+1)
        plt.imshow(obs[i])
    plt.show()

def generate_default_config():
    print('Generating default config file...')
    allconfigs = {}
    allconfigs['algo'] = return_algo_config()
    allconfigs['wrappers'] = return_wrapper_config()
    allconfigs['rewardfuncs'] = return_rewardfunc_config()
    return allconfigs

def create_file(args: argparse.Namespace):
    #
    time_str = datetime.now().strftime('%Y%m%d-%H:%M:%S')
    exp_name = f'{time_str}_{args.savename if args.savename is not None else args.scenario}'
    exp_root = os.path.join('./exp', exp_name)
    #
    video_dir = os.path.join(exp_root, 'video')
    log_dir = os.path.join(exp_root, 'logs')

    # Make sure the dirs exists
    Path(exp_root).mkdir(parents=True, exist_ok=True)
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # copy the config file to the exp folder
    os.system(f'cp {args.conf} {exp_root}')

    return exp_root, log_dir, video_dir
    


def main(args: argparse.Namespace):
    if args.epath: evaluation_client(args)

    # Create the experiment folder
    exp_root, log_dir, video_dir = create_file(args)
    
    # Load configuration
    with open(args.conf, 'r') as file:
        config = json.load(file)
    

    vec_eval_env = SubprocVecEnv([lambda: get_env(args, config) for _ in range(5)])
    vec_train_env = SubprocVecEnv([lambda: get_env(args, config) for _ in range(args.n_train)])

    print("vec_eval_ok")

    eval_callback = CustomEvalRewardShapingCallback(
        reward_shaping_func=get_rewardshaping_func(config['rewardfuncs']),
        eval_env=vec_eval_env,
        n_eval_episodes=args.n_eval,
        # eval_freq=5000,
        eval_freq=5000,
        log_path=log_dir,
        best_model_save_path=os.path.join(exp_root, 'best_eval_model'),
        deterministic=True
    )
    #
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(exp_root, 'checkpoints'),
        name_prefix="checkpoints",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    print("checkpoint_callback_ok")

    general_alg_kwargs = dict(
        env=vec_train_env,
        verbose=args.verbose,
        seed=args.seed,
        tensorboard_log=log_dir,
    )

    model = None

    if args.finetune_path is not None:
        # model = get_algo(config['algo'], general_alg_kwargs)
        model = get_algo(config['algo'])
        model = model.load(args.finetune_path, **general_alg_kwargs)
    else:
        model = get_algo(config['algo'], general_alg_kwargs)

    model.learn(total_timesteps=args.total,
                    log_interval=2,
                    progress_bar=True,
                    callback=CallbackList([eval_callback, checkpoint_callback]))

    model.save(os.path.join(exp_root, 'last_model'))



   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gen', action="store_true", help='Generate the default config file.')
    parser.add_argument('--scenario', type=str, help='The scenario to be used')
    parser.add_argument('--conf', type=str, help='The path to the configuration file.')

    # Optional
    parser.add_argument('--total', type=int, default=9999999999, help='The total number of timesteps to be trained.')
    parser.add_argument('--n-train', type=int, default=10, help='The number of training environments.')
    parser.add_argument('--n-eval', type=int, default=5, help='The number of evaluation environments.')

    parser.add_argument('--debug', type=str, default="", help='Whether to use debug mode.')
    parser.add_argument('--verbose', action='store_true', help='Whether to use verbose mode.')
    parser.add_argument('--seed', type=int, default=1, help='The seed to be used.')
    parser.add_argument('--savename', type=str, default=None, help='Name the folder')

    # ================== Evaluation ============================
    parser.add_argument('--eport',  type=str, default=None, help='The path to the model to be evaluated.')
    parser.add_argument('--epath', type=str, default=None, help='Pass the client ip address.')

    # ================== Fine Tune ============================
    parser.add_argument('--finetune-path', '--fp', type=str, default=None,
                        help='The path to the model to be fine tuned.')
    
    args = parser.parse_args()

    
    # generate the default config file
    if args.gen:
        config = generate_default_config()
        with open("config.json", 'w') as file:
            json.dump(config, file, indent=4)
        exit(0)

    # check the scenario and the config file is specified
    if args.scenario is None or args.conf is None:
        raise ValueError('Please specify the scenario and the config file.')

    # check the config file is exist
    if not os.path.exists(args.conf):
        raise ValueError(f'Config file {args.conf} does not exist.')
    
    # check the scenario is exist
    if args.debug == "env":
        debug(args)
        exit(0)


    main(args)
