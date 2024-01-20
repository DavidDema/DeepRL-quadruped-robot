import os.path
from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from rl_modules.actor_critic import ActorCritic
import matplotlib.pyplot as plt
import numpy as np
import time


class RLAgent(nn.Module):
    def __init__(self,
                 env: GOEnv,
                 storage: Storage,
                 actor_critic: ActorCritic,
                 lr=1e-3,
                 value_loss_coef=1.0,
                 num_batches=1,
                 num_epochs=1,
                 device='cpu',
                 action_scale=0.3,
                 ppo_eps=0.2,
                 entropy_coef=0.01,
                 use_clipped_value_loss=True,
                 schedule='adaptive', #"fixed",
                 desired_kl=0.01,
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
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        #ppo
        self.ppo_eps = ppo_eps
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.entropy_coef = entropy_coef
        self.lr = lr

    def act(self, obs):
        # Compute the actions and values
        action = self.actor_critic.act(obs).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()

        action_mean = self.actor_critic.action_mean.squeeze()
        self.transition.action_mean = action_mean.detach().cpu().numpy()
        action_sigma = self.actor_critic.action_std.squeeze()
        self.transition.action_sigma = action_sigma.detach().cpu().numpy()

        return self.transition.action

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

    def update(self, ppo=True):
        mean_value_loss = 0
        mean_actor_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device)

        for (obs_batch, actions_batch, target_values_batch, advantages_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, returns_batch) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            if ppo:
                value_batch = self.actor_critic.evaluate(obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
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
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.lr = min(1e-2, self.lr * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.lr

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps
                )
                actor_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.ppo_eps, self.ppo_eps
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            else:
                actor_loss = (-advantages_batch * actions_log_prob_batch).mean()
                value_loss = advantages_batch.pow(2).mean()
                loss = actor_loss + self.value_loss_coef * value_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_actor_loss += actor_loss.item()

        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    def play(self, is_training=True, early_termination=True):
        obs, _ = self.env.reset()
        infos = []
        for _ in range(self.storage.max_timesteps):
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                if is_training:
                    action = self.act(obs_tensor)
                else:
                    action = self.inference(obs_tensor)
            obs_next, reward, terminate, info = self.env.step(action*self.action_scale)
            infos.append(info)
            if is_training:
                self.store_data(obs, reward, terminate)
            if terminate and early_termination:
                obs, _ = self.env.reset()
            else:
                obs = obs_next
        if is_training:
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float())

        return infos

    def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=50):
        rewards_collection = []
        mean_value_loss_collection = []
        mean_actor_loss_collection = []

        for it in range(num_learning_iterations):
            # play games to collect data
            infos = self.play(is_training=True)
            # improve policy with collected data
            mean_value_loss, mean_actor_loss = self.update()

            rewards_collection.append(np.sum(self.storage.rewards))
            mean_value_loss_collection.append(mean_value_loss)
            mean_actor_loss_collection.append(mean_actor_loss)

            infos = self.play(is_training=False)

            info_mean = infos[0]
            for key in info_mean.keys():
                key_values = []
                for info in infos:
                    key_values.append(info[key])
                info_mean[key] = np.mean(key_values)

            print(f"------- Episode {it}/{num_learning_iterations} ------------")

            print("------ Losses ------ ")
            print(f"Critic loss              : {mean_value_loss}")
            print(f"Actor loss               : {mean_actor_loss}")

            print("------ Rewards ------ ")
            max_length = max(len(key) for key in info_mean.keys())
            for key in info_mean.keys():
                print(key.ljust(max_length) + "\t : " + str(info_mean[key]))
            print("--------------------- ")


            if it % num_steps_per_val == 0:
                self.plot_results(save_dir, it, num_learning_iterations, critic_losses=mean_value_loss_collection,
                                  actor_losses=mean_actor_loss_collection, rewards=rewards_collection)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    @staticmethod
    def plot_results(save_dir, it, num_learning_iterations, critic_losses, actor_losses, rewards):

        plt.clf()
        plt.plot(np.array(actor_losses) * 100, label='actor')
        plt.plot(np.array(critic_losses) / 50, label='critic')
        #plt.plot(np.array(rewards), label='reward')
        plt.title("Actor/Critic Loss (" + str(it) + "/" + str(num_learning_iterations) + ")")
        plt.ylabel("Loss")
        plt.xlabel("Episodes")
        plt.savefig(os.path.join(save_dir, f'ac_loss.png'))
        plt.legend()
        plt.draw()
        plt.pause(0.1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        torch.save(self.state_dict(), 'checkpoints/model.pt')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



