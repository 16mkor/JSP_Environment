import os
import datetime as dt

# from stable_baselines import TRPO
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from JSP_env.envs.production_env import ProductionEnv


def run(config, parameters):
    """
    This function is used to run a reinforcement learning model with various flags for loading, logging, saving,
    and rendering. The function first sets up the environment and model based on the specified parameters,
    then trains the model, saves, and evaluates it if specified.
    If the RENDER_FLAG is set to True, the function also renders the environment after training.
    :param config: the configuration of the experiment
    :param parameters: the configuration parameters of the environment of the experiment
    :return: no return value
    """

    """Set up Environment & Model"""
    env = _set_up_env(MULT_ENV_FLAG=config['MULT_ENV_FLAG'], parameter=parameters)
    model = _create_model(LOAD_FLAG=config['LOAD_FLAG'], load_path=config['load_path'], env=env, model_type=config['model_type'])

    """Set up Logger"""
    if config['LOGGER_FLAG']:
        logger = _set_up_logger(logging_path=config['logging_path'], model=model)

    """Train Model"""
    model.learn(total_timesteps=1)

    """Evaluate Model"""
    if config['EVAL_FLAG']:
        evaluate_policy(model=model, env=env, return_episode_rewards=False)

    """Save Model"""
    if config['SAVE_FLAG']:
        _save_model(save_path=config['save_path'], model_type=config['model_type'], model=model)

    """Render Model in Environemnt"""
    if config['RENDER_FLAG']:
        _render(env=env, model=model)


def _set_up_env(MULT_ENV_FLAG, parameters):
    """
    Creates either a vectorized environment, if the model will be trained on multiple environments,
    or a single environment.
    Bevor returning the environment, a check on the accordance to the gym specifications is made.
    :param MULTENV_FLAG: a flag to indicate whether to use multiple environments and therefore use 'DummyVecEnv'
    :param parameters: the configuration parameters of the environment of the experiment
    :return: the environment to train the Reinforcement Learning model
    """
    if MULT_ENV_FLAG:
        env = DummyVecEnv([lambda: ProductionEnv(parameters)])  # Vectorized Environment for multiple environments
    else:
        env = ProductionEnv(parameters)
    check_env(env)  # Check if Environment follows the structure of Gym. -> passed :)

    return env


def _set_up_logger(logging_path, model, log_config=["stdout", "csv", "tensorboard"]):
    """
    Creats a Stable-Baselines3 Logger and assigns it to the model.
    :param logging_path: the path to save the log files
    :param model: the Reinforcement Learning model
    :param log_config: format of logging -> Available formats are ["stdout", "csv", "log", "tensorboard", "json"]
    :return: the logger object
    """
    logger = configure(logging_path, log_config)
    model.set_logger(logger)  # Set new logger

    return logger


def _save_model(save_path, model_type, model):
    """
    Create the directory, if needed, and saves the model to that directory.
    :param save_path: the path to save the trained model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported. I used to
    create the folder structure if needed.
    :param model: the Reinforcement Learning model
    :return: no return value
    """
    if not os.path.exists(save_path):
        os.mkdirs(save_path + model_type)  # Create full path with needed sub folders
    model.save(save_path + model_type + '/' + model_type + dt.datetime.now().strftime('%Y_%m_%d_%Hh_%Mmin_%Ssec'))


def _load_model(load_path, model_type):
    """
    Loads the Reinforcement Learning model depending on the model type.
    :param load_path: the path to load a pre-existing model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported
    :return: the reinforcement learning model
    """
    if model_type == 'PPO':
        model = PPO.load(load_path)
    elif model_type == 'DQN':
        model = DQN.load(load_path)
    elif model_type == 'A2C':
        model = A2C.load(load_path)
    # ToDo: TRPO needs to be integrated
    # elif model_type == 'TRPO':
        # model = TRPO.load(load_path)
    else:
        print(model_type, 'not found!')

    return model


def _create_model(LOAD_FLAG, load_path, env, model_type):
    """
    Create or load the Reinforcement Learning model depending on the model_type.
    :param LOAD_FLAG: a flag to indicate whether to load a pre-existing model
    :param load_path: the path to load a pre-existing model
    :param env: the environment to train the reinforcement learning model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported
    :return: the reinforcement learning model
    """
    if LOAD_FLAG:
        model = _load_model(load_path, model_type)
    else:
        if model_type == 'PPO':
            model = PPO("MlpPolicy", env, verbose=1)
        elif model_type == 'DQN':
            model = DQN("MlpPolicy", env, verbose=1)
        elif model_type == 'A2C':
            model = A2C("MlpPolicy", env, verbose=1)
        # elif model_type == 'TRPO':
        # model = TRPO("MlpPolicy", env, verbose=1)
        else:
            print(model_type, 'not found!')

    return model


def _render(env, model):
    """
    Render the interaction of the agent in the environment. Currently not supported!
    :param env: the environment to train the reinforcement learning model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported
    :return: no return value
    """
    # TODO: Not visualy implemented yet
    obs = env.reset()
    for i in range(2_000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
