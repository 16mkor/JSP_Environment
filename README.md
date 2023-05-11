# JSP_Environment
Derived from the SimRLFab, as described below.

## Installation of JSP_Environemnt

Required packages (Python 3.9): 
```bash
pip3 install -r requirements.txt
```

## Run

```bash
python3 main.py 
```

### Prepared Configuration

**usage:** main.py [-h] [-rl_algorithm RL]  [-max_episode_timesteps T] [-num_episodes E] [-settings S]
               [-env_config C]
               
[-h, --help]              show this help message and exit      

[-rl_algorithm]           provide one of the RL algorithms: PPO, or DQN (default: PPO)

[-max_episode_timesteps]  provide the number of maximum timesteps per episode (default: 1000) 

[-num_episodes]           provide the number of episode (default: 1000)

[-settings]               provide the filename for the configuration of the settings of the Experiment as in config folder (default: NO_SETTINGS)

[-env_config]             provide the filename for the configuration of the environment as in config folder (default: NO_CONFIG)

## SimPyRLFab
Simulation and reinforcement learning framework for production planning and control of complex job shop manufacturing systems (https://github.com/AndreasKuhnle/SimRLFab).

### Introduction

Complex job shop manufacturing systems are motivated by the manufacturing characteristics of the semiconductor wafer fabrication. A job shop consists of several machines (processing resources) that process jobs (products, orders) based on a defined list or process steps. After every process, the job is dispatched and transported to the next processing machine. Machines are usually grouped in sub-areas by the type processing type, i.e. similar processing capabilities are next to each other. 

In operations management, two tasks are considered to improve opperational efficiency, i.e. increase capacity utilization, raise system throughput, and reduce order cycle times. First, job shop scheduling is an optimization problem which assigns a list of jobs to machines at particular times. It is considered as NP-hard due to the large number of constraints and even feasible solutions can be hard to compute in reasonable time. Second, order dispatching optimizes the order flow and dynamically determines the next processing resource. Depending on the degree of stochastic processes either scheduling or dispatching is enforced. In manufacturing environments with a high degree of unforseen and stochastic processes, efficient dispatching approaches are required to operate the manufacturing system on a robust and high performance. 

According to Mönch (2013) there are several characteristics that cause the complexity characteristics:
- Unequal job release dates
- Sequence-dependent setup times
- Prescribed job due dates
- Different process types (e.g., single processing, batch processing)
- Frequent machine breakdowns and other disturbances
- Re-entrant flows of jobs

> Mönch, L., Fowler, J. W., & Mason, S. J. (2013). Production Planning and Control for Semiconductor Wafer Fabrication Facilities.

This framework provides an integrated simulation and reinforcement learning model to investigate the potential of data-driven reinforcement learning in production planning and control of complex job shop systems. The simulation model allows parametrization of a broad range of job shop-like manufacturing systems. Furthermore, performance statistics and logging of performance indicators are provided. Reinforcement learning is implemented to control the order dispatching and several dispatchin heuristics provide benchmarks that are used in practice. 

### Features

The simulation model covers the following features (`initialize_env.py`):
- Number of resources:
    - Machines: processing resources
    - Sources: resources where new jobs are created and placed into the system
    - Sinks: resources where finished jobs are placed
    - Transport resources: dispatching and transporting resources
- Layout of fix resources (machines, sources, and sinks) based on a distance matrix
- Sources:
    - Buffer capacity (only outbound buffer)
    - Job release / generation process
    - Restrict jobs that are released at a specific source
- Machines:
    - Buffer capacity (inbound and outbound buffers)
    - Process time and distribution for stochastic process times
    - Machine group definition (machines in the same group are able to perform the same process and are interchangeable)
    - Breakdown process based on mean time between failure (MTBL) and mean time of line (MTOL) definition
    - Changeover times for different product variants
- Sinks:
    - Buffer capacity (only inbound buffer)
- Transport resources:
    - Handling times
    - Transport speed
- Others:
    - Distribution of job variants
    - Handling times to load and unload resources
    - Export frequency of log-files
    - Turn on / off console printout for detailed report of simulation processes and debugging
    - Seed for random number streams

<p align="center"> 
<img src="/docu/layout.png" width="500">
</p>

The reinforcement learning is based on the **Tensorforce** library and allows the combination of a variety of popular deep reinforcement learning models. Further details are found in the **Tensorforce** documentation. Problem-specific configurations for the order dispatching task are the following (`initialize_env.py`):
- State representation, i.e. which information elements are part of the state vector
- Reward function (incl. consideration of multiple objective functions and weighted reward functions according to action subset type)
- Action representation, i.e. which actions are allowed (e.g., "idling" action) and type of mapping of discrete action number to dispatching decisions
- Episode definition and limit
- RL-specific parameters such as learning rate, discount rate, neural network configuration etc. are defined in the Tensorforce agent configuration file

In practice, heuristics are applied to control order dispatching in complex job shop manufacturing systems. The following heuristics are provided as benchmark:
- **FIFO**: First In First Out selects the next job accoding to the sequence of appearance in the system to prevent long cycle times. 
- **NJF**: Nearest Job First dispatches the job which is closest to the machine to optimize the route of the dispatching / transport resource.
- **EMPTY**: It dispatches a job to the machine with the smalles inbound buffer to supply machines that run out of jobs.
The destination machine for the next process of a job, if alternative machines are available based on the machine group definition, is determined for all heuristics according to the smallest number of jobs in the inbound buffer.

By default, the sequencing and processing of orders at machines is based on a FIFO-procedure.

The default configuration provided in this package is based on a semiconductor setting presented in:
> Kuhnle, A., Röhrig, N., & Lanza, G. (2019). "Autonomous order dispatching in the semiconductor industry using reinforcement learning", Procedia CIRP, p. 391-396

> Kuhnle, A., Schäfer, L., Stricker, N., & Lanza, G. (2019). "Design, Implementation and Evaluation of Reinforcement Learning for an Adaptive Order Dispatching in Job Shop Manufacturing Systems". Procedia CIRP, p. 234-239.

### Running guide

Set up and run a simulation and training experiment:
1. Define production parameters and agent configuration (see above, `initialize_env.py`)
2. Set timesteps per episode and number of episodes (`run.py`)
3. Select RL-agent configuration file (`run.py`)
4. Run
5. Analyse performance in log-files

### Installation of the former SimPyRLFab

Required packages (Python 3.6): 
```bash
pip install -r requirements.txt
```

### Extensions (not yet implemented)

- Job due dates
- Batch processing
- Alternative maintenance strategies (predictive, etc.)
- Alternative strategies for order sequencing and processing at machines
- Mutliple RL-agents for several production control tasks
- etc.

## Acknowledgments

We extend our sincere thanks to the German Federal Ministry of Education and Research (BMBF) for supporting this research (reference nr.: 02P14B161).


#Regarding the GTrXL implementation:

# TransformerXL as Episodic Memory in Proximal Policy Optimization

This repository features a PyTorch based implementation of PPO using TransformerXL (TrXL). Its intention is to provide a clean baseline/reference implementation on how to successfully employ memory-based agents using Transformers and PPO.

# Features

- Episodic Transformer Memory
  - TransformerXL (TrXL)
  - Gated TransformerXL (GTrXL)
- Environments
  - Proof-of-concept Memory Task (PocMemoryEnv)
  - CartPole
    - Masked velocity
  - Minigrid Memory
    - Visual Observation Space 3x84x84
    - Egocentric Agent View Size 3x3 (default 7x7)
    - Action Space: forward, rotate left, rotate right
  - [MemoryGym](https://github.com/MarcoMeter/drl-memory-gym)
    - Mortar Mayhem
    - Mystery Path
    - Searing Spotlights (WIP)
- Tensorboard
- Enjoy (watch a trained agent play)

# Citing this work

```bibtex
@article{pleines2023trxlppo,
  title = {TransformerXL as Episodic Memory in Proximal Policy Optimization},
  author = {Pleines, Marco and Pallasch, Matthias and Zimmer, Frank and Preuss, Mike},
  journal= {Github Repository},
  year = {2023},
  url = {https://github.com/MarcoMeter/episodic-transformer-memory-ppo}
}
```

# Contents

- [Installation](#installation)
- [Train a model](#train-a-model)
- [Enjoy a model](#enjoy-a-model)
- [Episodic Transformer Memory Concept](#episodic-transformer-memory-concept)
- [Hyperparameters](#hyperparameters)
      - [Episodic Transformer Memory](#episodic-transformer-memory)
      - [General](#general)
      - [Schedules](#schedules)
- [Add Environment](#add-environment)
- [Tensorboard](#tensorboard)
- [Results](#results)

# Installation

Install [PyTorch](https://pytorch.org/get-started/locally/) 1.12.1 depending on your platform. We recommend the usage of [Anaconda](https://www.anaconda.com/).

Create Anaconda environment:
```bash
conda create -n transformer-ppo python=3.7 --yes
conda activate transformer-ppo
```

CPU:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

CUDA:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Install the remaining requirements and you are good to go:

```bash
pip install -r requirements.txt
```

# Train a model

The training is launched via `train.py`. `--config` specifies the path to the yaml config file featuring hyperparameters. The `--run-id` is used to distinguish training runs. After training, the trained model will be saved to `./models/$run-id$.nn`.

```bash
python train.py --config configs/minigrid.yaml --run-id=my-trxl-training
```

# Enjoy a model

To watch an agent exploit its trained model, execute `enjoy.py`. Some pre-trained models can be found in: `./models/`. The to-be-enjoyed model is specified using the `--model` flag.

```bash
python main.py --model=models/mortar_mayhem_grid_trxl.nn
```

# Episodic Transformer Memory Concept

![transformer-xl-model](./docs/assets/trxl_architecture.png)

# Hyperparameters

#### Episodic Transformer Memory

<table>
  <thead>
    <tr>
      <th>Hyperparameter</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>num_blocks</td>
      <td>Number of transformer blocks</td>
    </tr>
    <tr>
      <td>embed_dim</td>
      <td>Embedding size of every layer inside a transformer block</td>
    </tr>
    <tr>
      <td>num_heads</td>
      <td>Number of heads used in the transformer's multi-head attention mechanism</td>
    </tr>
    <tr>
      <td>memory_length</td>
      <td>Length of the sliding episodic memory window</td>
    </tr>
    <tr>
      <td>positional_encoding</td>
      <td>Relative and learned positional encodings can be used</td>
    </tr>
    <tr>
      <td>layer_norm</td>
      <td>Whether to apply layer normalization before or after every transformer component. Pre layer normalization refers to the identity map re-ordering.</td>
    </tr>
    <tr>
      <td>gtrxl</td>
      <td>Whether to use Gated TransformerXL</td>
    </tr>
    <tr>
      <td>gtrxl_bias</td>
      <td>Initial value for GTrXL's bias weight</td>
    </tr>    
  </tbody>
</table>

#### General

<table>
  <tbody>
    <tr>
      <td>gamma</td>
      <td>Discount factor</td>
    </tr>
    <tr>
      <td>lamda</td>
      <td>Regularization parameter used when calculating the Generalized Advantage Estimation (GAE)</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Number of cycles that the entire PPO algorithm is being executed</td>
    </tr>
    <tr>
      <td>n_workers</td>
      <td>Number of environments that are used to sample training data</td>
    </tr>
    <tr>
      <td>worker_steps</td>
      <td>Number of steps an agent samples data in each environment (batch_size = n_workers * worker_steps)</td>
    </tr>
    <tr>
      <td>epochs</td>
      <td>Number of times that the whole batch of data is used for optimization using PPO</td>
    </tr>
    <tr>
      <td>n_mini_batch</td>
      <td>Number of mini batches that are trained throughout one epoch</td>
    </tr>
    <tr>
      <td>value_loss_coefficient</td>
      <td>Multiplier of the value function loss to constrain it</td>
    </tr>
    <tr>
      <td>hidden_layer_size</td>
      <td>Number of hidden units in each linear hidden layer</td>
    </tr>
    <tr>
      <td>max_grad_norm</td>
      <td>Gradients are clipped by the specified max norm</td>
    </tr>
  </tbody>
</table>

#### Schedules

These schedules can be used to polynomially decay the learning rate, the entropy bonus coefficient and the clip range.

<table>
    <tbody>
    <tr>
      <td>learning_rate_schedule</td>
      <td>The learning rate used by the AdamW optimizer</td>
    </tr>
    <tr>
      <td>beta_schedule</td>
      <td>Beta is the entropy bonus coefficient that is used to encourage exploration</td>
    </tr>
    <tr>
      <td>clip_range_schedule</td>
      <td>Strength of clipping losses done by the PPO algorithm</td>
    </tr>
  </tbody>
</table>

# Add Environment

Follow these steps to train another environment:

1. Implement a wrapper of your desired environment. It needs the properties `observation_space`, `action_space` and `max_episode_steps`. The needed functions are `render()`, `reset()` and `step`.
2. Extend the `create_env()` function in `utils.py` by adding another if-statement that queries the environment's "type"
3. Adjust the "type" and "name" key inside the environment's yaml config

Note that only environments with visual or vector observations are supported. Concerning the environment's action space, it can be either discrte or multi-discrete.

# Tensorboard

During training, tensorboard summaries are saved to `summaries/run-id/timestamp`.

Run `tensorboad --logdir=summaries` to watch the training statistics in your browser using the URL [http://localhost:6006/](http://localhost:6006/).

# Results

Every experiment is repeated on 5 random seeds. Each model checkpoint is evaluated on 50 unknown environment seeds, which are repeated 5 times. Hence, one data point aggregates 1250 (5x5x50) episodes. Rliable is used to retrieve the interquartile mean and the bootstrapped confidence interval. The training is conducted using the more sophisticated DRL framework [neroRL](https://github.com/MarcoMeter/neroRL). The clean GRU-PPO baseline can be found [here](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt).

## Mystery Path Grid (Goal & Origin Hidden)

![mpg_off_results](./docs/assets/mpg_off.png)

TrXL and GTrXL have identical performance. See [Issue #7](https://github.com/MarcoMeter/episodic-transformer-memory-ppo/issues/7).

## Mortar Mayhem Grid (10 commands)

![mmg_10_results](./docs/assets/mmg_10.png)
