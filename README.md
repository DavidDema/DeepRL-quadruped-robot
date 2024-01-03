# deepRL_cheetah

## DONE:
- implement GAE: as discribed in ActorCritic slides
- implement PPO: ActorCritic slides, https://spinningup.openai.com/en/latest/algorithms/ppo.html
- plot and dynamically update Actor/Critic Loss
- use LTSM as hidden layer for network

## TODO:
- check implementations
- adapt reward-functions

## RESULTS:

### Actor/Critic losses without changes
![losseswithout changes](RL_Project/results/ac_loss.png)

### Actor/Critic losses without changes
![losses with GAE](RL_Project/results/ac_loss_gae.png)

### Actor/Critic losses with GAE and PPO
![losses with GAE and PPO](RL_Project/results/ac_loss_gae_ppo.png)

### Actor/Critic losses with GAE, PPO and LSTM
![losses with GAE, PPO and LSTM](RL_Project/results/ac_loss_gae_ppo_lstm.png)