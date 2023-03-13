import sys
from JSP_env.config import config
from JSP_env.run import run
import argparse

if __name__ == "__main__":
    """
    First get configuration of experiment and the environment will be extracted. 
    Then the experiment will be started.    
    """
    # get configuration of experiment and environment
    parser = argparse.ArgumentParser(description='Prepared Configuration')
    parser.add_argument('-rl_algorithm', metavar='RL', type=str, nargs=1, default=['PPO'],
                        help='provide one of the RL algorithms: PPO, TRPO, A2C, or DQN (default: PPO)')
    parser.add_argument('-max_episode_timesteps', metavar='T', type=int, nargs=1, default=[1_000],
                        help='provide the number of maximum timesteps per episode (default: 1000)')
    parser.add_argument('-num_episodes', metavar='E', type=int, nargs=1, default=[1_000],
                        help='provide the number of episode (default: 1000)')
    parser.add_argument('-settings', metavar='S', type=str, nargs=1, default=['NO_SETTINGS'],
                        help='provide the filename for the configuration of the settings of the Experiment'
                             ' as in config folder (default: NO_SETTINGS)')
    parser.add_argument('-env_config', metavar='C', type=str, nargs=1, default=['NO_CONFIG'],
                        help='provide the filename for the configuration of the environment as in config folder (default: NO_CONFIG)')
    args = parser.parse_args()

    exp_config = config.get_settings(args.settings[0], args.rl_algorithm[0])
    parameters = config.get_env_config(args.env_config[0])

    # Run the experiments with a certain configuration
    run(config=exp_config, parameters=parameters, timesteps=args.max_episode_timesteps[0], episodes=args.num_episodes[0])
