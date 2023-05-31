from JSP_Environments.config import config
from JSP_Environments.run import run
import argparse
from distutils.util import strtobool
import torch
import numpy as np

def parse_args():
    # get configuration of experiment and environment
    parser = argparse.ArgumentParser(description='Prepared Configuration')

    parser.add_argument("--gym-id", type=str, default="JSP_Environment",
                        help="the id of the gym environment")
    parser.add_argument('-rl', '--rl_algorithm', metavar='RL', type=str, nargs=1, default='PPO',
                        help='provide one of the RL algorithms: PPO, RecPPO, A2C, DQN, GTrXL-PPO, EMPTY, FIFO, RANDOM, '
                             'NJF (default: PPO)')
    parser.add_argument('-sc', '--scenario', metavar='SC', type=str, nargs=1, default='Basic',
                        help='Basic Scenario "Basic", Adjusted Scenario "Adjusted", Mixed Scenario "Mixed')
    parser.add_argument('-max_e', '--max_episode_timesteps', metavar='T', type=int, nargs=1, default=1_000,
                        help='provide the number of maximum timesteps per episode (default: 1_000)')
    parser.add_argument('-num_e', '--num_episodes', metavar='E', type=int, nargs=1, default=4_000,
                        help='provide the number of episode (default: 10000)')
    parser.add_argument('-se', '--seed', metavar='s', type=int, nargs=1, default=-1,
                        help='Seed for the pseudo random generators (default: random seed)')
    parser.add_argument('-s', '--settings', metavar='S', type=str, nargs=1, default='NO_SETTINGS',
                        help='provide the filename for the configuration of the settings of the Experiment'
                             ' as in config folder (default: NO_SETTINGS)')
    parser.add_argument('-conf', '--env_config', metavar='C', type=str, nargs=1, default='NO_CONFIG',
                        help='provide the filename for the configuration of the environment as in config folder '
                             '(default: NO_CONFIG)')
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument('-d', '--device', type=str, default='cpu', nargs=1,
                        help="if toggled, cuda will be enabled by default")

    args = vars(parser.parse_args())
    for arg in args:
        if type(args[arg]) == list:
            args[arg] = args[arg][0]
    if args['seed'] == -1:
        args['seed'] = np.random.randint(low=0, high=10 ** 5)
    if args['rl_algorithm'] == 'FIFO' or args['rl_algorithm'] == 'NJF' or \
            args['rl_algorithm'] == 'RANDOM' or args['rl_algorithm'] == 'EMTPY':
        args['num_episodes'] = 100

    return args


if __name__ == "__main__":
    """
    First get configuration of experiment and the environment will be extracted. 
    Then the experiment will be started.    
    """
    args = parse_args()
    exp_config = config.get_settings(args['settings'], args['rl_algorithm'])
    parameters = config.get_env_config(args['env_config'])
    # Run the RL-based experiments with a certain configuration
    run(config=exp_config, parameters=parameters, args=args)
