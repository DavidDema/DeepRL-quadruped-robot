import torch
import torch.nn as nn
from networks.networks import MLP
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 cfg
                 ):
        super().__init__()

        self.ac_cfg = cfg['ac']

        self.actor = MLP(dim_in=state_dim,
                         dim_hidden=self.ac_cfg['actor_hidden_dim'],
                         dim_out=action_dim,
                         n_layers=self.ac_cfg['actor_n_layers'],
                         act=nn.ELU(),
                         output_act=nn.Tanh(),
                         using_norm=False,
                         use_dropout=self.ac_cfg['use_dropout'],
                         dropout_p=self.ac_cfg['dropout_p'],
                         # use_ltsm=self.ac_cfg['use_ltsm'],
                         )

        self.critic = MLP(dim_in=state_dim,
                          dim_hidden=self.ac_cfg['critic_hidden_dim'],
                          dim_out=1,
                          n_layers=self.ac_cfg['critic_n_layers'],
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False,
                          use_dropout=self.ac_cfg['use_dropout'],
                          dropout_p=self.ac_cfg['dropout_p'],
                          #use_ltsm=self.ac_cfg['use_ltsm'],
                          )

        # Action distribution
        self.std = nn.Parameter(self.ac_cfg['init_std'] * torch.ones(action_dim))
        self.distribution = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        with torch.no_grad():
            actions_mean = self.actor(observations)

        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
