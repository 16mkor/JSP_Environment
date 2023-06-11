""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on an OpenAI Gym environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
from typing import Any
from typing import Dict
import os
import json
import gym
import optuna
import datetime as dt
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sb3_contrib import TRPO, RecurrentPPO
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from rl_zoo3 import linear_schedule

from JSP_Environments.gtrxl_ppo.trainer import PPOTrainer


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            # activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    # subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

    net_arch = {"small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }
    return hyperparams


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_recppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_gtrxl_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for GTrXL hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    lambda_ = trial.suggest_float("max_grad_norm", 0.5, 1.0, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_mini_batch = trial.suggest_int("n_mini_batch", 1, 8, log=True)
    value_loss_coefficient = trial.suggest_float("value_loss_coefficient", 0.1, 1.0, log=True)
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 64, 256, log=True)
    # transformer
    num_blocks = trial.suggest_int("num_blocks", 1, 8, log=True)
    num_heads = trial.suggest_categorical("num_heads", [2 ** 1, 2 ** 2, 2 ** 3])
    embed_dim = trial.suggest_categorical("embed_dim", [2 ** 5, 2 ** 6, 2 ** 7])

    memory_length = trial.suggest_int("memory_length", 16, 64, log=True)

    return {
        "environment":
            {
                "type": "JSP"
            },
        "gamma": gamma,
        "lamda": lambda_,
        "updates": 300,
        "epochs": 1000,
        "n_workers": 2,
        "worker_steps": 256,
        "n_mini_batch": n_mini_batch,
        "value_loss_coefficient": value_loss_coefficient,
        "hidden_layer_size": hidden_layer_size,
        "max_grad_norm": max_grad_norm,
        "transformer":
            {
                "num_blocks": num_blocks,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "memory_length": 32,
                "positional_encoding": "",
                "layer_norm": "pre",
                "gtrxl": True,
                "gtrxl_bias": 0.0
            },
        "learning_rate_schedule":
            {
                "initial": learning_rate,
                "final": learning_rate * 10e-2,
                "power": 1.0,
                "max_decay_steps": 300
            },
        "beta_schedule":
            {
                "initial": 0.001,
                "final": 0.0001,
                "power": 1.0,
                "max_decay_steps": 300
            },
        "clip_range_schedule":
            {
                "initial": 0.2,
                "final": 0.2,
                "power": 1.0,
                "max_decay_steps": 300
            }
    }


def sample_trpo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # line_search_shrinking_factor = trial.suggest_categorical("line_search_shrinking_factor", [0.6, 0.7, 0.8, 0.9])
    n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
    cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
    # cg_damping = trial.suggest_categorical("cg_damping", [0.5, 0.2, 0.1, 0.05, 0.01])
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        # "cg_damping": cg_damping,
        "cg_max_steps": cg_max_steps,
        # "line_search_shrinking_factor": line_search_shrinking_factor,
        "n_critic_updates": n_critic_updates,
        "target_kl": target_kl,
        "learning_rate": learning_rate,
        "gae_lambda": gae_lambda,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def _hyperparameter_tuning(env, model_type):
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)
    N_TRIALS = 25
    N_STARTUP_TRIALS = 25
    N_EVALUATIONS = 10
    N_TIMESTEPS = int(5e5)
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    N_EVAL_EPISODES = 10

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
            sample_params = sample_trpo_params(trial)
            model = TRPO(env=env, policy='MlpPolicy', **sample_params)
        elif model_type == 'PPO':
            sample_params = sample_ppo_params(trial)
            model = PPO(env=env, policy='MlpPolicy', **sample_params)
        elif model_type == 'RecPPO':
            sample_params = sample_recppo_params(trial)
            model = RecurrentPPO(env=env, policy='MlpLstmPolicy', **sample_params)
        elif model_type == 'GTrXL_PPO':
            sample_params = sample_gtrxl_params(trial)
            trainer = PPOTrainer(sample_params, run_id='GTrXL-PPO-v2')

        # Create env used for evaluation.
        eval_env = env  # Monitor(ProductionEnv(parameter, seed, time_steps, num_episodes, model_type))
        # Create the callback that will periodically evaluate and report the performance.
        eval_callback = TrialEvalCallback(
            eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
        )

        nan_encountered = False
        try:
            if model_type != 'GTrXL_PPO':
                model.learn(N_TIMESTEPS, callback=eval_callback)
            else:
                _ = trainer.run_training()
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN.
            print(e)
            nan_encountered = True
        finally:
            # Free memory.
            if model_type != 'GTrXL-PPO':
                model.env.close()
            else:
                trainer.close()
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

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 10,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
