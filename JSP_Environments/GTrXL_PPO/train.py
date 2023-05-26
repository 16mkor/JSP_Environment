import torch
import numpy as np
import pickle
import gym
from JSP_Environments.GTrXL_PPO.model import ActorCriticModel
from JSP_Environments.GTrXL_PPO.utils import init_transformer_memory, create_env
from JSP_Environments.GTrXL_PPO.trainer import PPOTrainer
from JSP_Environments.GTrXL_PPO.yaml_parser import YamlParser
from stable_baselines3.common.monitor import Monitor


def train() ->None:
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help

    Options:
        --config=<path>            Path to the yaml config file [default: ./config/poc_memory_env.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """

    run_id = "GTrXL-PPO-v2"
    cpu = True
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser('JSP_Environments/config/production_env.yaml').get_config()

    # Determine the device to be used for training and set the default tensor type
    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    path = trainer.run_training()
    GTrXL_experiment(model_path=path, device=device)
    trainer.close()


def GTrXL_experiment(model_path, device):
    # Set inference device and default tensor type
    device = torch.device(device)
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Initialize env
    dummy_env = Monitor(create_env(config["environment"]))
    env = gym.wrappers.RecordEpisodeStatistics(dummy_env)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    model.load_state_dict(state_dict)
    model.to(device)
    # model.train()
    model.eval()

    # Run and render episode

    memory, memory_mask, memory_indices = init_transformer_memory(config["transformer"], env.max_episode_steps, device)
    memory_length = config["transformer"]["memory_length"]
    epoch_rewards = []

    for episode in range(config['epochs']):
        done = False
        episode_rewards = []
        obs = env.reset()
        t = 0
        while not done:
            # Prepare observation and memory
            obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
            in_memory = memory[0, memory_indices[t].unsqueeze(0)]
            t_ = max(0, min(t, memory_length - 1))
            mask = memory_mask[t_].unsqueeze(0)
            indices = memory_indices[t].unsqueeze(0)
            # Forward model
            policy, value, new_memory = model(obs, in_memory, mask, indices)
            memory[:, t] = new_memory
            # Sample action
            action = []
            for action_branch in policy:
                action.append(action_branch.sample().item())
            # Step environemnt
            obs, reward, done, info = env.step(action[0])
            episode_rewards.append(reward)
            t += 1
        epoch_rewards.append(episode_rewards)

    # print("Episode length: " + str(info["length"]))
    # print("Episode reward: " + str(info["reward"]))

    env.close()


if __name__ == "__main__":
    train()
