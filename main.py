from JSP_Environments.config import config
from JSP_Environments.run import run
import argparse

if __name__ == "__main__":
    """
    First get configuration of experiment and the environment will be extracted. 
    Then the experiment will be started.    
    """
    # get configuration of experiment and environment
    parser = argparse.ArgumentParser(description='Prepared Configuration')
    parser.add_argument('-rl', '--rl_algorithm', metavar='RL', type=str, nargs=1, default='PPO',
                        help='provide one of the RL algorithms: PPO, A2C, DQN, GTrXL-PPO, EMPTY, FIFO, RANDOM, '
                             'NJF (default: PPO)')
    parser.add_argument('-max_e', '--max_episode_timesteps', metavar='T', type=int, nargs=1, default=1_000,
                        help='provide the number of maximum timesteps per episode (default: 1_000)')
    parser.add_argument('-num_e', '--num_episodes', metavar='E', type=int, nargs=1, default=10_000,
                        help='provide the number of episode (default: 5000)')
    parser.add_argument('-se', '--seed', metavar='s', type=int, nargs=1, default=10,
                        help='Seed for the pseudo random generators (default: 10)')
    parser.add_argument('-da', '--dataset', metavar='d', type=str, nargs=1, default='test_trajectories.pkl',
                        help='Seed for the pseudo random generators (default: 10)')
    parser.add_argument('-s', '--settings', metavar='S', type=str, nargs=1, default='NO_SETTINGS',
                        help='provide the filename for the configuration of the settings of the Experiment'
                             ' as in config folder (default: NO_SETTINGS)')
    parser.add_argument('-conf', '--env_config', metavar='C', type=str, nargs=1, default='NO_CONFIG',
                        help='provide the filename for the configuration of the environment as in config folder '
                             '(default: NO_CONFIG)')
    parser.add_argument('-d', '--device', metavar='dev', type=str, nargs=1, default="cpu",
                        help='run on cpu oder gpu (cuda)? Default: cpu')
    args = vars(parser.parse_args())

    if type(args['rl_algorithm']) == list:
        args['rl_algorithm'] = args['rl_algorithm'][0]
        args['device'] = args['device'][0]

    exp_config = config.get_settings(args['settings'], args['rl_algorithm'])
    parameters = config.get_env_config(args['env_config'])

    # Run the RL-based experiments with a certain configuration
    run(config=exp_config, parameters=parameters, args=args)
