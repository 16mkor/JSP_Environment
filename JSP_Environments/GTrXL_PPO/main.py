import numpy as np
import pickle
import torch

# from docopt import docopt
from JSP_Environments.GTrXL_PPO.model import ActorCriticModel
from JSP_Environments.GTrXL_PPO.utils import create_env


def init_transformer_memory(trxl_conf, max_episode_steps, device):
    """Returns initial tensors for the episodic memory of the transformer.

    Arguments:
        trxl_conf {dict} -- Transformer configuration dictionary
        max_episode_steps {int} -- Maximum number of steps per episode
        device {torch.device} -- Target device for the tensors

    Returns:
        memory {torch.Tensor}, memory_mask {torch.Tensor}, memory_indices {torch.Tensor} -- Initial episodic memory, episodic memory mask, and sliding memory window indices
    """
    # Episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
    # Episdic memory tensor
    memory = torch.zeros((1, max_episode_steps, trxl_conf["num_blocks"], trxl_conf["embed_dim"])).to(device)
    # Setup sliding memory window indices
    repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0),
                                          trxl_conf["memory_length"] - 1, dim=0).long()
    memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in
                                  range(max_episode_steps - trxl_conf["memory_length"] + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices


def GTrXL_experiment(env, config):
    # Set inference device and default tensor type
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    # state_dict, config = pickle.load(open(model_path, "rb"))

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    # model.load_state_dict(state_dict)
    model.to(device)
    model.train()
    # model.eval()

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

    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

    env.close()
