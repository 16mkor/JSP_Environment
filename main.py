import sys
import os

from JSP_env.config import config
from JSP_env.run import run


if __name__ == "__main__":
    # get configuration of experiment and environment
    config = config.get_exp_config(sys.argv[1])
    parameters = config.get_env_config(sys.argv[2])

    # Run the experiments with a certain configuration
    run(config=config, parameters=parameters)
