import os.path
from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from rl_modules.actor_critic import ActorCritic
import matplotlib.pyplot as plt
import numpy as np

class RLAgent(nn.Module):
    def __init__(self,
                 env: GOEnv,
                 storage: Storage,
                 actor_critic: ActorCritic,
                 lr=1e-2, ## 1e-3 Ausgangswert
                 value_loss_coef=1.0,
                 num_batches=1,
                 num_epochs=1,
                 device='cpu',
                 action_scale=2.0, # 0.3 Ausgangswert
                 ppo_eps=0.2, # 0.2 Ausgangswert
                 target_kl=0.5,
                 
                 desired_kl=0.01, ##Kira PPO## ##0.01
                 learning_rate=1e-2,
                 use_clipped_value_loss=True,
                 entropy_coef=0.001,
                 schedule="adaptive", # fixed
                 ):
        super().__init__()
        self.env = env
        self.storage = storage
        self.actor_critic = actor_critic
        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.value_loss_coef = value_loss_coef
        self.device = device
        self.action_scale = action_scale
        self.transition = Storage.Transition()
        # create the normalizer
        # Der Adam-Optimierer ist eine Variante des stochastischen Gradientenabstiegs (SGD), der adaptive Lernraten für jedes Gewicht bereitstellt und dazu beiträgt, das Training von neuronalen Netzwerken zu verbessern.
        # lr=lr: Dieser Parameter legt die Lernrate für den Adam-Optimierer fest. Die Lernrate steuert die Größe der Schritte, die der Optimierer während des Trainings geht.
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # ppo
        self.ppo_eps = ppo_eps
        self.target_kl = target_kl

        ##Kira PPO##
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate
        self.use_clipped_value_loss = use_clipped_value_loss
        self.entropy_coef = entropy_coef
        self.schedule = schedule


        # Epsilon-Greedy
        self.exploration_prob = 0.9

    def act(self, obs):
        # Compute the actions and values
        action = self.actor_critic.act(obs, exploration_prob=self.exploration_prob).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()
        
        ##Kira PPO##
        action_mean = self.actor_critic.action_mean.squeeze()
        self.transition.action_mean = action_mean.detach().cpu().numpy()
        action_sigma = self.actor_critic.action_std.squeeze()
        self.transition.action_sigma = action_sigma.detach().cpu().numpy()

        return self.transition.action

    '''
     inference: Sie gibt die vorhergesagten Aktionen basierend auf den Beobachtungen zurück, 
     ohne die Verteilung zu aktualisieren.
    '''
    def inference(self, obs):
        return self.actor_critic.act_inference(obs).squeeze().detach().cpu().numpy()

    def store_data(self, obs, reward, done):
        self.transition.obs = obs
        self.transition.reward = reward
        self.transition.done = done

        # Record the transition
        self.storage.store_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, last_obs):
        last_values = self.actor_critic.evaluate(last_obs).detach().cpu().numpy()
        return self.storage.compute_returns(last_values)

    def update(self, ppo=False, ppo_eth=True):
        mean_value_loss = 0
        mean_actor_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device) # get data from storage

        '''
        for obs_batch, actions_batch, target_values_batch, advantages_batch, actions_log_prob_old_batch in generator:
            self.actor_critic.act(obs_batch, exploration_prob=self.exploration_prob) # evaluate policy
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            
            # compute losses
            if ppo:
                policy_ratio = torch.exp(actions_log_prob_batch - actions_log_prob_old_batch)
                policy_ratio_clipped = policy_ratio.clamp(1 - self.ppo_eps, 1 + self.ppo_eps)
                actor_loss = -torch.min(policy_ratio * advantages_batch, policy_ratio_clipped * advantages_batch).mean() 

                policy_kl = np.abs(policy_ratio.mean().detach().cpu().numpy()) - 1
                if policy_kl >= self.target_kl:
                    print("ppo early termination: policy_ratio: " + str(policy_kl))
                    break
            else:     
                actor_loss = (-advantages_batch * actions_log_prob_batch).mean()

            critic_loss = advantages_batch.pow(2).mean()
            loss = actor_loss + self.value_loss_coef * critic_loss
        '''
            
            
            ##Kira PPO##
        for (obs_batch, actions_batch, target_values_batch, advantages_batch, actions_log_prob_old_batch, old_mu_batch,
             old_sigma_batch, returns_batch) in generator:
            # Evaluate policy
            self.actor_critic.act(obs_batch, exploration_prob=self.exploration_prob)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            if ppo_eth:
                value_batch = self.actor_critic.evaluate(obs_batch)

                mu_batch = self.actor_critic.action_mean        ## mean -value
                sigma_batch = self.actor_critic.action_std      ## standad-deviation
                entropy_batch = self.actor_critic.entropy       ## Unsicherheitsberechnung

                # KL -> Kullback-Leibler-Divergenz
                '''
                Berechnung der KL-Divergenz (kl) zwischen der aktuellen Politik und der vorherigen Politik. Dies geschieht unter Verwendung der Logarithmusregel 
                der Wahrscheinlichkeit und der Formel für die KL-Divergenz zweier normalverteilter Funktionen.
                Berechnung des Durchschnitts (kl_mean) der KL-Divergenz über die Aktionen im Batch.
                Dynamische Anpassung der Lernrate basierend auf dem Durchschnitt der KL-Divergenz:
                Wenn kl_mean mehr als das Zweifache des gewünschten KL-Werts beträgt, wird die Lernrate verringert.
                Wenn kl_mean weniger als die Hälfte des gewünschten KL-Werts und größer als null ist, wird die Lernrate erhöht.
                Die Lernratenanpassung erfolgt durch Veränderung des Lernratenparameters der Optimierung (self.optimizer.param_groups).
                '''
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                '''
                Der Surrogatverlust ist eine Fehlermetrik, die den Unterschied 
                zwischen der aktuellen Politik 
                und der aktualisierten Politik misst. 
                '''
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                '''
                Verhältnis, das zwischen 1.0 - self.clip_param und 1.0 + self.clip_param geklemmt ist. 
                Dieser Schritt dient dazu, den Einfluss von großen Änderungen in der Politik zu begrenzen.
                '''
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps
                )
                '''
                Der endgültige Aktorverlust wird als der größere der beiden Verluste (surrogate und surrogate_clipped) genommen, 
                und dann wird der Durchschnitt über den Batch berechnet.
                '''
                actor_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                '''
                - value_batch: Die vorhergesagten Werte (Wertefunktion) für die Zustände, die vom Netzwerk vorhergesagt wurden.
                - target_values_batch: Die Zielwerte (Vergleichswerte) für die Zustände
                - returns_batch: Die tatsächlichen Rückgaben (Gesamtrückgaben) für die Zustände
                '''
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.ppo_eps, self.ppo_eps
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                '''
                Wenn self.use_clipped_value_loss wahr ist, wird ein geklippter Werteverlust verwendet. 
                Dies kann dazu dienen, zu verhindern, dass der Werteverlust zu stark variiert.
                value_loss: Der maximale der beiden Losses wird ausgewählt und gemittelt.
                ------------
                Die Gesamtverlustfunktion loss setzt sich aus diesen Teilen zusammen:
                - actor_loss: Surrogatverlust für die Politik.
                - self.value_loss_coef * value_loss: Der gewichtete Wertverlust.
                - self.entropy_coef * entropy_batch.mean(): Der negative gewichtete Entropieverlust.
                '''
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            elif ppo:
                policy_ratio = torch.exp(actions_log_prob_batch - actions_log_prob_old_batch)
                policy_ratio_clipped = policy_ratio.clamp(1 - self.ppo_eps, 1 + self.ppo_eps)
                actor_loss = -torch.min(policy_ratio * advantages_batch, policy_ratio_clipped * advantages_batch).mean()

                policy_kl = np.abs(policy_ratio.mean().detach().cpu().numpy()) - 1
                if policy_kl >= self.target_kl:
                    print("ppo early termination: policy_ratio: " + str(policy_kl))
                    break

                value_loss = advantages_batch.pow(2).mean()
                loss = actor_loss + self.value_loss_coef * value_loss

            else:
                actor_loss = (-advantages_batch * actions_log_prob_batch).mean()

                value_loss = advantages_batch.pow(2).mean()
                loss = actor_loss + self.value_loss_coef * value_loss

            # Gradient step - update the parameters
            self.optimizer.zero_grad()  ## Der Gradient des vorherigen Schritts wird zurückgesetzt, um sicherzustellen, dass keine akkumulierten Gradienten vorhanden sind.
            loss.backward()             ## Der Backpropagation-Schritt, bei dem der Gradient der Verlustfunktion bezüglich der Netzwerkparameter berechnet wird.
            self.optimizer.step()       ## Der Optimierungsschritt, bei dem die Netzwerkparameter anhand der berechneten Gradienten und der gewählten Optimierungsalgorithmus (Adam in diesem Fall) aktualisiert werden.

            mean_value_loss += value_loss.item()    ## Akkumuliere den Wert des Werteverlusts für die Durchschnittsberechnung am Ende des Trainings.
            mean_actor_loss += actor_loss.item()    ## Akkumuliere den Wert des Surrogatverlusts für die Durchschnittsberechnung am Ende des Trainings.
            
        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates  ## Teilt die summierten Value Loss-Werte durch die Gesamtanzahl der Updates (num_updates), um den durchschnittlichen Value Loss pro Update zu berechnen.
        mean_actor_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    def play(self, is_training=True, early_termination=True):
        last_termination_timestep = 0

        obs, _ = self.env.reset() # first reset env
        infos = []
        for t in range(self.storage.max_timesteps): # rollout an episode ##(YAN)## increase max_timestepsase max_timesteps
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                if is_training:
                    action = self.act(obs_tensor) # sample an action from policy
                else:
                    action = self.inference(obs_tensor)
            self.env.timestep = t-last_termination_timestep
            self.env.max_timesteps = self.storage.max_timesteps - last_termination_timestep
            obs_next, reward, terminate, info = self.env.step(delta_q=action*self.action_scale) # perform one step action
            infos.append(info)
            if is_training:
                self.store_data(obs, reward, terminate) # collect data to storage
            if terminate and early_termination:
                obs, _ = self.env.reset()
                last_termination_timestep = t
            else:
                obs = obs_next
        if is_training:
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float()) # Prepare data for update, e.g., advantage estimate

        return infos

    #def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=50, num_plots=10): (Ausgangswerte)
    def learn(self, save_dir, num_learning_iterations=2000, num_steps_per_val=50, num_plots=10): ##(YAN)## increase num_learning_iterations
        rewards_collection = []
        mean_value_loss_collection = []
        mean_actor_loss_collection = []
        value_loss_init = 0
        actor_loss_init = 0
        reward_init = 0

        plt.ion()
        plt.show(block=False)
        
        for it in range(1, num_learning_iterations + 1):

            # Exploration probability should be high at the beginning and low with well learned network
            progress = it/(num_learning_iterations+1)
            max_prob = 1
            min_prob = 0.3     ## 0.05
            #self.exploration_prob = -(max_prob-min_prob)*progress + max_prob
            k = 4 ## (Ausgangswert 4)
            self.exploration_prob = np.exp(-(k*progress-np.log(max_prob-min_prob)))+min_prob
            #self.exploration_prob = max_prob
            print(f"Exploration prob: {self.exploration_prob}")


            # play games to collect data
            infos = self.play(is_training=True) # play
            # improve policy with collected data
            mean_value_loss, mean_actor_loss = self.update() # update

            if it == 1:
                value_loss_init = mean_value_loss
                actor_loss_init = mean_actor_loss
                reward_init = np.mean(self.storage.rewards)

            #rewards_collection.append((np.mean(self.storage.rewards)+reward_init)/reward_init)
            rewards_collection.append(np.mean(self.storage.rewards))
            mean_value_loss_collection.append(mean_value_loss/value_loss_init)
            mean_actor_loss_collection.append(mean_actor_loss/actor_loss_init)

            self.plot_results(save_dir, infos, rewards_collection, mean_actor_loss_collection, mean_value_loss_collection, it, num_learning_iterations, num_plots)

            if it % num_steps_per_val == 0:
                infos = self.play(is_training=False)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    @staticmethod
    def plot_results(save_dir, infos, rewards, actor_losses, critic_losses, it, num_learning_iterations, num_plots):

        if it % num_plots == 0:
            plt.clf()
            plt.plot(np.array(actor_losses), label='actor')
            plt.plot(np.array(critic_losses), label='critic')
            plt.plot(np.array(rewards), label='reward')
            plt.title("Actor/Critic Loss (" + str(it) + "/" + str(num_learning_iterations) + ")")
            plt.ylabel("Loss")
            plt.xlabel("Episodes")
            plt.savefig(os.path.join(save_dir, f'ac_loss.png'))
            plt.legend()
            plt.draw()
            plt.grid(True)
            plt.ylim([-0.5, 1.5])
            plt.pause(0.1)
        '''
        print(f"------- Episode {it}/{num_learning_iterations} ------------")
        #print(f"Exploration prob         : {self.exploration_prob}")
        print("Losses:")
        print(f"Critic loss              : {critic_losses[-1]}")
        print(f"Actor loss               : {actor_losses[-1]}")
        
        
        

        info_mean = infos[0]
        for key in info_mean.keys():
            key_values = []
            for info in infos:
                key_values.append(info[key])
            info_mean[key] = np.mean(key_values)

        plot_all_rewards = True
        if plot_all_rewards:
            print("Rewards:")
            for key in info_mean.keys():
                print(key + "\t : " + str(info_mean[key]))
        else:
            print(f"Avg.Reward               : {info_mean['total_reward']}")


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        torch.save(self.state_dict(), 'checkpoints/model.pt')
        print("Saved model parameters to checkpoints/model.pt")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    '''
        
        keys = ['track_vel_reward',
                'living_reward',
                'pitchroll_rate_reward',
                'orient_reward',
                'pitchroll_reward',
                'yaw_rate_reward',
                'healthy_reward',
                'z_pos_reward',
                'z_vel_reward',
                'feet_slip',
                'forward'
                ]
        infos_array = np.array([[info[key] for key in keys] for info in infos])
        mean_values = np.mean(infos_array, axis=0)

        keys_print = ['track_vel_reward      : ',
                      'pitchroll_rate_reward : ',
                      'orient_reward         : ',
                      'pitchroll_reward      : ',
                      'yaw_rate_reward       : ',
                      'healthy_reward        : ',
                      'living_reward         : ', 
                      'z_pos_reward          : ',
                      'z_vel_reward          : ',
                      'feet_slip             : ',
                      'forward               : ',
                      ]
        
        print(f"------- Episode {it}/{num_learning_iterations} ------------")
        print("Losses:")
        print(f"Critic loss              : {critic_losses[-1]}")
        print(f"Actor loss               : {actor_losses[-1]}")

        print("--------- Rewards ---------")
        for i, key in enumerate(keys_print):
            print(key + str(mean_values[i]))


    def save_model(self, path):
            torch.save(self.state_dict(), path)
            torch.save(self.state_dict(), 'checkpoints/model.pt')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



