# Reinforcement Learning: Group 5 (384.195 Robot Learning)

## Team Members

- Kira Herlemann
- Nino Wegleitner
- Piet
- David Demattio

## Code structure
Only relevant files are described in the following tree structure:
```
deeprl_cheetah
│   test.py
│   train.py
│   plot.py
│   README.md
│   requirements.txt
│
└───checkpoints
│   │
│   └───model             # Contains the files of the latest training process !
│       │   config.yaml   # latest configuration file used
│       │   data.pkl      # latest data collection for plotting
│       │   model.pt      # latest model parameter file
│   │
│   └───2024-01-14        # other training session
│   │
│   └───2024-01-15
│   .
│   . 
│   .
│   
└───rl_modules
│   │   actor_critic.py   # MLP instances for actor and critic
│   │   storage.py        # Storage class (save data, GAE)
│   │   rl_agent.py       # Agent class (PPO, play, learn, ..)
│
└───env
│   │   go_env.py
│   └───go
│       │   go1.xml         
│       │   go1_torque.xml  
│       │   scene.xml         # scene that calls go1.xml file
│       │   scene_torque.xml  # scene that calls go1_torque.xml file 
│                             # (used for torque control with config.yaml->use_torque:true)
│
└───networks
│   │   networks.py     # MLP Network Definition
│
└───config
│   │   config.yaml
│   │   config_handler.py # Config-Class is used for handling the "config.yaml"-File
```

## Usage
### Using the `train.py`-file

For model training you should set up the properties in the `config/config.yaml`. This file contains settings regaring the environment, reinforcement learning algorithm, networks and reward function. It is copied to the checkpoint location `checkpoints/YYYY-MM-DD/HH-MM-SS/model/`.\
During the training process the model parameters and relevant training data are saved in the location `checkpoints/.../model/`, this folder is copied to the location `checkpoints/latest` and represents the **latest training data**. 

### Using the `test.py`-file

The test-file can be used with a specific experiment (`use_experiment=True`), where the experiment folder has to be in the location `checkpoints/experiment_name/` and the variable `experiment=experiment_name` has to be defined.\
It is also possible to select a specific training set (`use_experiment=False`) as implemented in the original test.py-file.

## Project configuration and data logging

The config file "config/config.yaml" is copied into the checkpoint folder `checkpoint/.../model`.
In addition, the recorded relevant data values in rl_agend.learn() are saved as a dictionary in the same folder with a copy of the newest `model.pt` of the current training process.

## Results: (TODO)

### Actor/Critic losses without changes
![losses without changes](RL_Project/results/ac_loss.png)

### Actor/Critic losses without changes
![losses with GAE](RL_Project/results/ac_loss_gae.png)

### Actor/Critic losses with GAE and PPO
![losses with GAE and PPO](RL_Project/results/ac_loss_gae_ppo.png)

### Actor/Critic losses with GAE, PPO and LSTM
![losses with GAE, PPO and LSTM](RL_Project/results/ac_loss_gae_ppo_lstm.png)

### Actor/Critic losses with GAE, PPO, LSTM and adapted reward functions
%![losses with GAE, PPO, LSTM and adapted reward](RL_Project/results/ac_loss_gae_ppo_lstm_reward.png)

## TODO 

- check implementations
- calculate feet position
- add experience replay

### Results

- PPO on/off
- GAE on/off
- Longer Training Time
- Different Rewards
- Longer MaxTimesteps
- ?

## References

- ETH:
  * Paper: https://arxiv.org/pdf/2109.11978.pdf
  * Rewards: https://github.com/leggedrobotics/legged_gym/blob/20f7f92e89865084b958895d988513108b307e6c/legged_gym/envs/base/legged_robot.py#L853
  * PPO: https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
  * GAE: https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/storage/rollout_storage.py
- Nature Paper: https://www.nature.com/articles/s41598-023-38259-7
- use LTSM as hidden layer for network: 
    * https://github.com/Kaixhin/ACER/tree/master
- PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Reward-functions: 
    * https://www.nature.com/articles/s41598-023-38259-7 
    * https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=y79PoJOCIl-O
