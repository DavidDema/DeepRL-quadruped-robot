import torch
import torch.nn as nn
from networks.networks import MLP
from torch.distributions import Normal

import numpy as np

class ActorCritic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=1024,   # 512 (Ausgangswert)
                 n_layers=4,       #2 (Ausgangswert)
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
        self.distribution = Normal(mean, self.std)  ## Die update_distribution-Methode muss keinen expliziten Rückgabewert haben, da sie self.distribution aktualisiert, und diese aktualisierte Verteilung von den anderen Methoden verwendet wird.


    '''
    Diese Methode wird verwendet, um im Trainingsmodus Aktionen zu sampeln. 
    Es ruft self.update_distribution auf, um die Verteilung zu aktualisieren, 
    und verwendet dann diese Verteilung, um eine Aktion zu sampeln.
    ''' 
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    
    '''
    Für Reinforcement-Learning-Modelle, wie das Actor-Critic-Modell, 
    bedeutet Inferenz, dass das trainierte Modell (der Actor) verwendet wird, 
    um Aktionen basierend auf den aktuellen Beobachtungen zu generieren, ohne 
    die Gewichtungen des Modells dabei zu aktualisieren.
    '''
    def act_inference(self, observations):
        with torch.no_grad():   ## ohne Gradienten zu berechnen
            actions_mean = self.actor(observations)

        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
