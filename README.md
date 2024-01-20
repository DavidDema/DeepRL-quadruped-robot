# deepRL_cheetah

## DONE:
- implement GAE: as discribed in ActorCritic slides
- implement PPO: ActorCritic slides, https://spinningup.openai.com/en/latest/algorithms/ppo.html
- plot and dynamically update Actor/Critic loss and rewards
- use LTSM as hidden layer for network: 
    * https://github.com/Kaixhin/ACER/tree/master
- adapt reward-functions: 
    * https://www.nature.com/articles/s41598-023-38259-7 
    * https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=y79PoJOCIl-O
- change to ETH Zürich implementation:
    * https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
    * https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/modules/actor_critic.py
    * https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/runners/on_policy_runner.py
    * https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/storage/rollout_storage.py
    * https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
    * https://arxiv.org/pdf/2109.11978.pdf

## TODO:
- check implementations
- calculate feet position
- add experience replay

## RESULTS:

### Actor/Critic losses without changes
![losses without changes](RL_Project/results/ac_loss.png)

### Actor/Critic losses without changes
![losses with GAE](RL_Project/results/ac_loss_gae.png)

### Actor/Critic losses with GAE and PPO
![losses with GAE and PPO](RL_Project/results/ac_loss_gae_ppo.png)

### Actor/Critic losses with GAE, PPO and LSTM
![losses with GAE, PPO and LSTM](RL_Project/results/ac_loss_gae_ppo_lstm.png)

### Actor/Critic losses with GAE, PPO, LSTM and adapted reward functions
![losses with GAE, PPO, LSTM and adapted reward](RL_Project/results/ac_loss_gae_ppo_lstm_reward.png)