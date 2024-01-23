# deepRL_cheetah
with torque as SIM input (nino3 with action input)

## DONE:
- implement GAE: as discribed in ActorCritic slides
- implement PPO: ActorCritic slides, https://spinningup.openai.com/en/latest/algorithms/ppo.html
- plot and dynamically update Actor/Critic loss and rewards
- use LTSM as hidden layer for network: 
    * https://github.com/Kaixhin/ACER/tree/master
- adapt reward-functions: 
    * https://www.nature.com/articles/s41598-023-38259-7 
    * https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb#scrollTo=y79PoJOCIl-O

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