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
    parser.add_argument('-settings', metavar='set', type=str, nargs=1, default=['NO_SETTINGS'],
                        help='provide the filename for the configuration of the settings of the Experiment'
                             ' as in config folder')
    parser.add_argument('-env_config', metavar='model', type=str, nargs=1, default=['NO_CONFIG'],
                        help='provide the filename for the configuration of the environment as in config folder')
    args = parser.parse_args()

    exp_config = config.get_settings(args.settings[0])
    parameters = config.get_env_config(args.env_config[0])

    # Run the experiments with a certain configuration
    run(config=exp_config, parameters=parameters)
