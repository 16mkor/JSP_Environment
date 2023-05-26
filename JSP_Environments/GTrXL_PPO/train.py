import torch
from JSP_Environments.GTrXL_PPO.trainer import PPOTrainer


def training(env, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(env, config, run_id="run", device=device)
    trainer.run_training()
    trainer.close()

