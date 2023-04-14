import sys
from JSP_Environments.config import config
from JSP_Environments.run import run
from Online_Decision_Transformer.main import odt_experiment
import argparse

if __name__ == "__main__":
    """
    First get configuration of experiment and the environment will be extracted. 
    Then the experiment will be started.    
    """
    # get configuration of experiment and environment
    parser = argparse.ArgumentParser(description='Prepared Configuration')
    parser.add_argument('-rl', '--rl_algorithm', metavar='RL', type=str, nargs=1, default='PPO',
                        help='provide one of the RL algorithms: PPO, EMPTY, FIFO, RANDOM, NJF, A2C, or DQN (default: PPO)')
    parser.add_argument('-max_e', '--max_episode_timesteps', metavar='T', type=int, nargs=1, default=1_000,
                        help='provide the number of maximum timesteps per episode (default: 1_000)')
    parser.add_argument('-num_e', '--num_episodes', metavar='E', type=int, nargs=1, default=5_000,
                        help='provide the number of episode (default: 5000)')
    parser.add_argument('-se', '--seed', metavar='s', type=int, nargs=1, default=10,
                        help='Seed for the pseudo random generators (default: 10)')
    parser.add_argument('-da', '--dataset', metavar='d', type=str, nargs=1, default='test_trajectories.pkl',
                        help='Seed for the pseudo random generators (default: 10)')
    parser.add_argument('-s', '--settings', metavar='S', type=str, nargs=1, default='NO_SETTINGS',
                        help='provide the filename for the configuration of the settings of the Experiment'
                             ' as in config folder (default: NO_SETTINGS)')
    parser.add_argument('-conf', '--env_config', metavar='C', type=str, nargs=1, default='NO_CONFIG',
                        help='provide the filename for the configuration of the environment as in config folder (default: NO_CONFIG)')
    args = vars(parser.parse_args())
    if type(args['rl_algorithm']) == list:
        args['rl_algorithm'] = args['rl_algorithm'][0]
    exp_config = config.get_settings(args['settings'], args['rl_algorithm'])
    parameters = config.get_env_config(args['env_config'])
    if args['rl_algorithm'] != 'ODT':
        # Run the RL-based experiments with a certain configuration
        run(config=exp_config, parameters=parameters, timesteps=args['max_episode_timesteps'],
            seed=args['seed'], episodes=args['num_episodes'])
    else:
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--seed", type=int, default=10)
        parser.add_argument("--env", type=str, default="JSP_Env")

        # model options
        parser.add_argument("--K", type=int, default=20)
        parser.add_argument("--embed_dim", type=int, default=512)
        parser.add_argument("--n_layer", type=int, default=4)
        parser.add_argument("--n_head", type=int, default=4)
        parser.add_argument("--activation_function", type=str, default="relu")
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--eval_context_length", type=int, default=5)
        # 0: no pos embedding others: absolute ordering
        parser.add_argument("--ordering", type=int, default=0)

        # shared evaluation options
        parser.add_argument("--eval_rtg", type=int, default=3600)
        parser.add_argument("--num_eval_episodes", type=int, default=10)

        # shared training options
        parser.add_argument("--init_temperature", type=float, default=0.1)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
        parser.add_argument("--warmup_steps", type=int, default=10000)

        # pretraining options
        parser.add_argument("--max_pretrain_iters", type=int, default=1)
        parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

        # finetuning options
        parser.add_argument("--max_online_iters", type=int, default=1500)
        parser.add_argument("--online_rtg", type=int, default=7200)
        parser.add_argument("--num_online_rollouts", type=int, default=1)
        parser.add_argument("--replay_size", type=int, default=1000)
        parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
        parser.add_argument("--eval_interval", type=int, default=10)

        # environment options
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
        parser.add_argument("--save_dir", type=str, default="./exp")
        parser.add_argument("--exp_name", type=str, default="default")

        args = parser.parse_args()

        odt_experiment(args, exp_config, parameters)
