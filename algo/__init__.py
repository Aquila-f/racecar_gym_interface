# register your algo here
from .example_ppo.example_ppo import ExamplePPO
from .custom_ppo.custom_ppo import CustomPPO


def get_algo(algo_config: dict, general_alg_kwargs: dict = None):
    classname = algo_config['algo']
    cls = globals().get(classname, None)
    if cls is None:
        raise ValueError(f'Cannot find class "{classname}" in policy.')
    if general_alg_kwargs is None: return cls

    kwargs = algo_config['parameters']
    policy = cls(**kwargs, **general_alg_kwargs)
    return policy






def return_algo_config():
    config = {}
    config['algo'] = "# custom algo"
    config['parameters'] = {
        'policy': "CnnPolicy / MlpPolicy / MultiInputPolicy",
        'policy_kwargs': {
            'normalize_images': False,
            'ortho_init': False,
        },
        'n_steps': 10240,
        'n_epochs': 3,
        'batch_size': 512,
        'clip_range': 0.2,
        'ent_coef': 0.001,
        'learning_rate': 8e-5,
        'gamma': 0.997

    }
    return config
