import torch
import torch.nn as nn
from networks.networks import MLP
from torch.distributions import Normal

import numpy as np

class ActorCritic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=512,
                 n_layers=1, #2
                 init_std=1.0
                 ):
        super().__init__()
        self.actor = MLP(dim_in=state_dim,
                         dim_hidden=hidden_dim,
                         dim_out=action_dim,
                         n_layers=n_layers,
                         act=nn.ELU(),
                         output_act=nn.Tanh(),
                         using_norm=False)

        self.critic = MLP(dim_in=state_dim,
                          dim_hidden=hidden_dim,
                          dim_out=1,
                          n_layers=n_layers,
                          act=nn.ELU(),
                          output_act=None,
                          using_norm=False)

        # Action distribution
        self.std = nn.Parameter(init_std * torch.ones(action_dim))
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

    def act(self, observations, exploration_prob, **kwargs):
        """
        Generate a random sample from the updated distribution
        :param observations: Current state
        :param exploration_prob:
        :param kwargs:
        :return:
        """

        self.update_distribution(observations)
        actions_rand = self.distribution.sample()
        actions_opt = self.act_inference(observations)

        e = np.random.choice([1, 2], 1, p=[(1 - exploration_prob), exploration_prob])[0]
        if e == 1:
            actions = actions_opt
        else:
            actions = actions_rand

        return actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        with torch.no_grad():
            actions_mean = self.actor(observations)

        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
