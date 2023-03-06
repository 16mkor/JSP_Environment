import sys
from JSP_env.config import config
from JSP_env.run import run

if __name__ == "__main__":
    """
    First get configuration of experiment and the environment will be extracted. 
    Then the experiment will be started.    
    """
    # get configuration of experiment and environment
    exp_config = config.get_exp_config(sys.argv[1])
    parameters = config.get_env_config(sys.argv[2])

    # Run the experiments with a certain configuration
    run(config=exp_config, parameters=parameters)
