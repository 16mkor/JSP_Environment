import datetime as dt
import json
import os
import gym
import optuna
import torch
import torch.nn as nn

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from JSP_Environments.envs.hyperparameter_tuning import sample_a2c_params, sample_ppo_params, sample_dqn_params, \
    sample_recppo_params, TrialEvalCallback
from JSP_Environments.envs.production_env import ProductionEnv
from JSP_Environments.gtrxl_ppo.train import train, GTrXL_experiment


def run(config, parameters, args):
    """
    This function is used to run a reinforcement learning model with various flags for loading, logging, saving,
    and rendering. The function first sets up the environment and model based on the specified parameters,
    then trains the model, saves, and evaluates it if specified.
    If the RENDER_FLAG is set to True, the function also renders the environment after training.
    :param timesteps:
    :param episodes:
    :param config: the configuration of the experiment
    :param parameters: the configuration parameters of the environment of the experiment
    :return: no return value
    """
    timesteps = args['max_episode_timesteps']
    seed = args['seed']
    episodes = args['num_episodes']
    device = args['device']

    if config['model_type'] == 'GTrXL-PPO':
        """Start gtrxl_ppo Experiments"""
        train()

    else:
        """
        Start Experiments of PPO, RecPPO, DQN, A2C & Heuristics 
        """

        """Set up Environment"""
        env = _set_up_env(MULT_ENV_FLAG=config['MULT_ENV_FLAG'], parameter=parameters,
                          seed=seed, time_steps=timesteps, num_episodes=episodes, model_type=config['model_type'])

        if config['HYPERPARAM_FLAG']:
            """Do Hyperparamter Tuning and create model with tuned hyperparameter"""
            hyperparameter = _hyperparameter_tuning(env, config['model_type'], device=device)
            model = _create_model(LOAD_FLAG=config['LOAD_FLAG'], load_path=config['load_path'], env=env,
                                  model_type=config['model_type'], timesteps=timesteps, device=device, seed=seed,
                                  tensorboard_log_path=config['tensorboard_log'], hyperparam=hyperparameter)

        elif not config['HYPERPARAM_FLAG']:
            """Do Hyperparamter Tuning and create model with default hyperparameter"""
            model = _create_model(LOAD_FLAG=config['LOAD_FLAG'], load_path=config['load_path'], env=env,
                                  model_type=config['model_type'], timesteps=timesteps, device=device, seed=seed,
                                  tensorboard_log_path=config['tensorboard_log'])

        """Set up Logger"""
        if config['LOGGER_FLAG']:
            logger = _set_up_logger(logging_path=config['logging_path'], model=model)

        elif model != "Heuristic":
            """Train Model"""
            print('################################')
            print('### START TRAINING PROCEDURE ###')
            print('################################')
            model.learn(total_timesteps=(timesteps * episodes),
                        tb_log_name=config['model_type'])

            """Evaluate Model"""
            if config['EVAL_FLAG']:
                print('##################################')
                print('### START Evaluation PROCEDURE ###')
                print('##################################')
                evaluate_policy(model=model, env=env, return_episode_rewards=False)

            """Save Model"""
            if config['SAVE_FLAG']:
                print('Model saved!')
                _save_model(save_path=config['save_path'], model_type=config['model_type'], model=model)

            """Render Model in Environemnt"""
            if config['RENDER_FLAG']:
                _render(env=env, model=model)

        else:
            start = True
            for _ in range(episodes):
                time_steps = 0
                done = False  # terminated, truncated = False, False
                _ = env.reset()
                while not done:  # (terminated or truncated):
                    if config['model_type'] == 'RANDOM' or start == True:
                        action = env.action_space.sample()
                        start = False
                    else:
                        action = env.env.resources['transps'][0].next_action[0]
                    state, reward, done, info = env.step(action)
                    time_steps += 1


def _set_up_env(MULT_ENV_FLAG, parameter, seed, time_steps, num_episodes, model_type):
    """
    Creates either a vectorized environment, if the model will be trained on multiple environments,
    or a single environment.
    Bevor returning the environment, a check on the accordance to the gym specifications is made.
    :param MULTENV_FLAG: a flag to indicate whether to use multiple environments and therefore use 'SubprocVecEnv'
    :param parameters: the configuration parameters of the environment of the experiment
    :return: the environment to train the Reinforcement Learning model
    """
    if MULT_ENV_FLAG:
        env = SubprocVecEnv([lambda: Monitor(ProductionEnv(parameter, seed, time_steps, num_episodes,
                                                           model_type))])  # Vectorized Environment for multiple environments
    else:
        env = Monitor(ProductionEnv(parameter, seed, time_steps, num_episodes, model_type))
    env = gym.wrappers.RecordEpisodeStatistics(env)
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


def _load_model(load_path, model_type, device, seed, tensorboard_log_path):
    """
    Loads the Reinforcement Learning model depending on the model type.
    :param load_path: the path to load a pre-existing model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported
    :return: the reinforcement learning model
    """
    if model_type == 'PPO':
        model = PPO.load(load_path, device=device, tensorboard_log=tensorboard_log_path)
    if model_type == 'RecPPO':
        model = RecurrentPPO.load(load_path, device=device, tensorboard_log=tensorboard_log_path)
    elif model_type == 'DQN':
        model = DQN.load(load_path, device=device, tensorboard_log=tensorboard_log_path)
    elif model_type == 'A2C':
        model = A2C.load(load_path, device=device, tensorboard_log=tensorboard_log_path)
    elif model_type == 'TRPO':
        model = TRPO.load(load_path, device=device, tensorboard_log=tensorboard_log_path)
    elif model_type == "FIFO" or model_type == "NJF" or model_type == "EMPTY" or model_type == 'RANDOM':
        model = "Heuristic"
    else:
        print(model_type, 'not found!')

    if type(model) != str:
        model.seed = seed
    return model


def _create_model(LOAD_FLAG, load_path, env, model_type, timesteps, device, seed, tensorboard_log_path,
                  config=None, hyperparam=None):
    """
    Create or load the Reinforcement Learning model depending on the model_type.
    :param LOAD_FLAG: a flag to indicate whether to load a pre-existing model
    :param load_path: the path to load a pre-existing model
    :param env: the environment to train the reinforcement learning model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported
    :return: the reinforcement learning model
    """

    if LOAD_FLAG:
        model = _load_model(load_path, model_type, device, seed, tensorboard_log_path)
    else:
        if hyperparam != None:
            if hyperparam['net_arch'] == "small":
                arch = {"pi": [64, 64], "vf": [64, 64]}
            elif hyperparam['net_arch'] == "medium":
                arch = {"pi": [256, 256], "vf": [256, 256]}
            policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=arch)

            if model_type == 'PPO':
                model = PPO("MlpPolicy", env, verbose=0,
                            tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs,
                            n_steps=hyperparam["n_steps"], batch_size=hyperparam["batch_size"],
                            gamma=hyperparam["gamma"], learning_rate=hyperparam["learning_rate"],
                            ent_coef=hyperparam["ent_coef"], clip_range=hyperparam["clip_range"],
                            gae_lambda=hyperparam["gae_lambda"], max_grad_norm=hyperparam["max_grad_norm"],
                            vf_coef=hyperparam["vf_coef"])
            if model_type == 'RecPPO':
                model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, device=device,
                                     tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs,
                                     n_steps=hyperparam["n_steps"], batch_size=hyperparam["batch_size"],
                                     gamma=hyperparam["gamma"], learning_rate=hyperparam["learning_rate"],
                                     ent_coef=hyperparam["ent_coef"], clip_range=hyperparam["clip_range"],
                                     gae_lambda=hyperparam["gae_lambda"], max_grad_norm=hyperparam["max_grad_norm"],
                                     vf_coef=hyperparam["vf_coef"])
            elif model_type == 'DQN':
                policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=arch['pi'])
                model = DQN("MlpPolicy", env, verbose=1, device=device,
                            tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs,
                            gamma=hyperparam["gamma"], learning_rate=hyperparam["learning_rate"],
                            batch_size=hyperparam["batch_size"], buffer_size=hyperparam["buffer_size"],
                            train_freq=hyperparam["train_freq"],
                            exploration_fraction=hyperparam["exploration_fraction"],
                            exploration_final_eps=hyperparam["exploration_final_eps"],
                            target_update_interval=hyperparam["target_update_interval"],
                            learning_starts=hyperparam["learning_starts"])
            elif model_type == 'A2C':
                model = A2C("MlpPolicy", env, verbose=1, device=device,
                            tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs,
                            n_steps=hyperparam["n_steps"], gamma=hyperparam["gamma"],
                            gae_lambda=hyperparam["gae_lambda"], learning_rate=hyperparam["learning_rate"],
                            ent_coef=hyperparam["ent_coef"], normalize_advantage=hyperparam["normalize_advantage"],
                            max_grad_norm=hyperparam["max_grad_norm"], use_rms_prop=hyperparam["use_rms_prop"],
                            vf_coef=hyperparam["vf_coef"])
            elif model_type == 'TRPO':
                model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_path,
                             policy_kwargs=policy_kwargs, n_steps=hyperparam["n_steps"],
                             batch_size=hyperparam["batch_size"], gamma=hyperparam["gamma"],
                             cg_max_steps=hyperparam["cg_max_steps"], n_critic_updates=hyperparam["n_critic_updates"],
                             target_kl=hyperparam["target_kl"], learning_rate=hyperparam["learning_rate"],
                             gae_lambda=hyperparam["gae_lambda"])
            elif model_type == "FIFO" or model_type == "NJF" or model_type == "EMPTY" or model_type == 'RANDOM':
                model = "Heuristic"
            else:
                print(model_type, 'not found!')

        else:
            policy_kwargs = {}  # dict(activation_fn=nn.Tanh, net_arch = {"pi": [64, 64], "vf": [64, 64]})
            if model_type == 'PPO':
                model = PPO("MlpPolicy", env, verbose=1, n_steps=timesteps, device=device,
                            tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs)
            if model_type == 'RecPPO':
                model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, n_steps=timesteps, device=device,
                                     tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs)
            elif model_type == 'DQN':
                model = DQN("MlpPolicy", env, verbose=1, device=device,
                            tensorboard_log=tensorboard_log_path)
            elif model_type == 'A2C':
                model = A2C("MlpPolicy", env, verbose=1, n_steps=timesteps, device=device,
                            tensorboard_log=tensorboard_log_path, policy_kwargs=policy_kwargs)
            elif model_type == 'TRPO':
                model = TRPO("MlpPolicy", env, verbose=1, n_steps=timesteps, tensorboard_log=tensorboard_log_path,
                             policy_kwargs=policy_kwargs)
            elif model_type == "FIFO" or model_type == "NJF" or model_type == "EMPTY" or model_type == 'RANDOM':
                model = "Heuristic"
            else:
                print(model_type, 'not found!')
    if type(model) != str:
        model.seed = seed
    return model


def _render(env, model):
    """
    Render the interaction of the agent in the environment. Currently not supported!
    :param env: the environment to train the reinforcement learning model
    :param model_type: the type of the model being used. Currently, PPO, DQN, A2C and TRPO are supported
    :return: no return value
    """
    # TODO: Not visualy implemented yet
    obs, info = env.reset()
    for i in range(2_000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()


def _hyperparameter_tuning(env, model_type, device):
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)
    N_TRIALS = 50
    N_STARTUP_TRIALS = 35
    N_EVALUATIONS = 24
    N_TIMESTEPS = int(5e5)
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    N_EVAL_EPISODES = 50

    """N_TRIALS = 1
    N_STARTUP_TRIALS = 1
    N_EVALUATIONS = 1
    N_TIMESTEPS = int(1)
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    N_EVAL_EPISODES = 1"""

    DEFAULT_HYPERPARAMS = {}

    def objective(trial: optuna.Trial) -> float:
        kwargs = DEFAULT_HYPERPARAMS.copy()
        # Sample hyperparameters.
        if model_type == 'A2C':
            sample_params = sample_a2c_params(trial)
            model = A2C(env=env, policy='MlpPolicy', **sample_params)
        elif model_type == 'DQN':
            sample_params = sample_dqn_params(trial)
            model = DQN(env=env, policy='MlpPolicy', **sample_params)
        elif model_type == 'TRPO':
            sample_params = sample_dqn_params(trial)
            model = TRPO(env=env, policy='MlpPolicy', **sample_params)
        elif model_type == 'PPO':
            sample_params = sample_ppo_params(trial)
            model = PPO(env=env, policy='MlpPolicy', **sample_params)
        elif model_type == 'RecPPO':
            sample_params = sample_recppo_params(trial)
            model = RecurrentPPO(env=env, policy='MlpLstmPolicy', **sample_params)

        # Create env used for evaluation.
        eval_env = env  # Monitor(ProductionEnv(parameter, seed, time_steps, num_episodes, model_type))
        # Create the callback that will periodically evaluate and report the performance.
        eval_callback = TrialEvalCallback(
            eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
        )

        nan_encountered = False
        try:
            model.learn(N_TIMESTEPS, callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)
            nan_encountered = True
        finally:
            # Free memory.
            if model_type != 'GTrXL-PPO':
                model.env.close()
            eval_env.close()

        # Tell the optimizer that the trial failed.
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.best_mean_reward

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    path = "JSP_Environments/log/Hyperparameter/"
    file = path + model_type + '_config_parameters' + "_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + '.json'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(file, 'a') as fp:
        json.dump(dict(trial.params.items()), fp)

    return dict(trial.params.items())
