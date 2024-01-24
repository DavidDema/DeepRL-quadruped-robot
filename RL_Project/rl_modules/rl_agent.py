import os.path
import time

from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from rl_modules.actor_critic import ActorCritic
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
import pickle
import shutil

class RLAgent(nn.Module):
    def __init__(self,
                 env: GOEnv,
                 storage: Storage,
                 actor_critic: ActorCritic,
                 cfg,
                 device='cpu',
                 ppo_eps=0.2, # eigenes PPO
                 target_kl=0.5, # eigenes PPO
                 ):
        super().__init__()
        self.env = env
        self.storage = storage
        self.actor_critic = actor_critic
        self.cfg = cfg
        self.alg_cfg = cfg['algorithm']
        self.runner_cfg = cfg['runner']

        self.value_loss_coef = self.alg_cfg['value_loss_coef']
        self.clip_param = self.alg_cfg['clip_param']
        self.use_clipped_value_loss = self.alg_cfg['use_clipped_value_loss']
        self.desired_kl = self.alg_cfg['desired_kl']
        self.entropy_coef = self.alg_cfg['entropy_coef']
        self.gamma = self.alg_cfg['gamma']
        self.lam = self.alg_cfg['lam']
        self.learning_rate = self.alg_cfg['learning_rate']
        self.max_grad_norm = self.alg_cfg['max_grad_norm']

        self.num_batches = self.alg_cfg['num_batches']
        self.num_epochs = self.alg_cfg['num_epochs']
        self.schedule = self.alg_cfg['schedule']
        self.action_scale = self.alg_cfg['action_scale']

        self.device = device

        self.transition = Storage.Transition()
        # create the normalizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        self.use_ppo = self.alg_cfg['use_ppo']
        # für eigenes ppo
        self.ppo_eps = ppo_eps
        self.target_kl = target_kl

    def act(self, obs):
        # Compute the actions and values
        action = self.actor_critic.act(obs).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()

        self.transition.action_mean = self.actor_critic.action_mean.detach().cpu().numpy()
        self.transition.action_std = self.actor_critic.action_std.detach().cpu().numpy()

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
        return self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, own_ppo=False):
        mean_value_loss = 0
        mean_actor_loss = 0
        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device) # get data from storage

        for (obs_batch, actions_batch, target_values_batch, advantages_batch,
             old_actions_log_prob_batch, returns_batch, old_mu_batch, old_sigma_batch) in generator:
            self.actor_critic.act(obs_batch) # evaluate policy
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # compute losses
            if self.use_ppo:
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
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                #TODO: nn.utils2.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_actor_loss += surrogate_loss.item()

            else:
                if own_ppo:
                    policy_ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
                    policy_ratio_clipped = policy_ratio.clamp(1 - self.ppo_eps, 1 + self.ppo_eps)
                    actor_loss = -torch.min(policy_ratio * advantages_batch, policy_ratio_clipped * advantages_batch).mean()

                    policy_kl = np.abs(policy_ratio.mean().detach().cpu().numpy()) - 1
                    if policy_kl >= self.target_kl:
                        print("ppo early termination: policy_ratio: " + str(policy_kl))
                        break
                else:
                    # TODO: check implementation -> actor_loss always = 1.0
                    actor_loss = (-advantages_batch * actions_log_prob_batch).mean()

                critic_loss = advantages_batch.pow(2).mean()
                loss = actor_loss + self.value_loss_coef * critic_loss

                # Gradient step - update the parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mean_value_loss += critic_loss.item()
                mean_actor_loss += actor_loss.item()

        num_updates = self.num_epochs * self.num_batches
        mean_value_loss /= num_updates
        mean_actor_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_actor_loss

    def play(self, is_training=True, early_termination=True):
        last_termination_timestep = 0
        obs, _ = self.env.reset() # first reset env

        infos = []
        for t in range(self.storage.max_timesteps): # rollout an episode
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                if is_training:
                    action = self.act(obs_tensor) # sample an action from policy
                else:
                    action = self.inference(obs_tensor)

            # timesteps since last termination
            self.env.timestep = t-last_termination_timestep
            # remaining episode time since termination
            self.env.max_timesteps = self.storage.max_timesteps - last_termination_timestep

            obs_next, reward, terminate, info = self.env.step(action*self.action_scale) # perform one step action
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

    def learn(self, save_dir):

        num_learning_iterations = self.runner_cfg['max_iterations']
        save_interval = self.runner_cfg['save_interval']
        plot_interval = self.runner_cfg['plot_interval']
        save_data = self.runner_cfg['save_data']
        save_model = self.runner_cfg['save_model']
        max_timesteps = self.runner_cfg['max_timesteps']
        first_save_iter = self.runner_cfg['first_save_iter']

        # Create data dictionary
        data = defaultdict(list)
        data['num_learning_iterations'].append(num_learning_iterations)

        plt.ion()
        plt.show(block=False)
        start = time.time()
        for it in range(1, num_learning_iterations + 1):
            start_iter = time.time()
            progress = it / num_learning_iterations
            print("---------------")
            print(f"Episode {it}/{num_learning_iterations} ({progress*100:.1f}%)(Runtime {(time.time()-start)/60:.1f}min)")

            # play games to collect data
            infos = self.play(is_training=True)  # play
            # improve policy with collected data
            mean_value_loss, mean_actor_loss = self.update()  # update
            print(f"Actor Loss\t\t: {mean_actor_loss:.4f}")
            print(f"Value Loss\t\t: {mean_value_loss:.2f}")

            infos_array = infos[0]
            info_rewards = infos[0]['rewards']
            info_mean = infos[0]
            for key in info_mean.keys():
                key_values = []
                if key == "rewards":
                    continue
                    for info in infos:
                        for rew_key in info[key]:
                            # TODO
                            pass

                for info in infos:
                    key_values.append(info[key])
                infos_array[key] = key_values
                info_mean[key] = np.mean(key_values)

            # Append relevant data to the dict for postprocessing
            data['actor_loss'].append(mean_actor_loss)
            data['critic_loss'].append(mean_value_loss)
            data['reward'].append(np.mean(self.storage.rewards))
            data['reward_sum'].append(np.sum(self.storage.rewards))
            data['traverse'].append(info_mean['traverse'])
            data['side'].append(info_mean['side'])
            data['height'].append(info_mean['height'])
            data['yaw'].append(info_mean['yaw'])
            data['roll'].append(info_mean['roll'])
            data['pitch'].append(info_mean['pitch'])
            data['time'].append(time.time()-start)
            data['epsiode'].append(it)

            # plot
            if it % plot_interval == 0:
                save_dir_model = os.path.join(save_dir, "model")
                self.plot_results(save_dir_model, data, savefig=True, running_plot=True)

            # Print Rewards
            if False:
                # all rewards
                print("------ Rewards ------ ")
                max_length = max(len(key) for key in info_mean.keys())
                for key in info_mean.keys():
                    print(key.ljust(max_length) + "\t : " + str(info_mean[key]))
            else:
                # total reward
                print(f"Total Reward\t: {info_mean['total_reward']:.2f}")

            # save model and data to checkpoints/.../model/
            if it % save_interval == 0 and it >= first_save_iter:
                save_dir_model = os.path.join(save_dir, "model")
                if save_model:
                    self.save_model(save_dir, f"{it}.pt")
                    print("Model saved")

                if save_data:
                    RLAgent.save_data(save_dir_model, data)
                    print("Data saved to checkpoints/.../data.pkl")

                if not os.path.exists("checkpoints/latest"):
                    os.makedirs("checkpoints/latest")
                shutil.copytree(save_dir_model, "checkpoints/latest", dirs_exist_ok=True)

            # Episode finished
            print(f"Episode finished after {time.time()-start_iter:.2f}s")

        plt.show()
        # Learning finished
        print(f"Training finished after {(time.time()-start)/60} minutes!")

    @staticmethod
    def plot_results(save_dir, data, savefig=False, running_plot=False):

        # Scaling
        scale_actor = 0.05
        scale_critic = 5000
        scale_reward = 10
        scale_meters = 1

        if running_plot:
            plt.clf()
        else:
            plt.figure()
        plt.plot(np.array(data['actor_loss'])/scale_actor, label=f'actor (x{scale_actor})')
        plt.plot(np.array(data['critic_loss'])/scale_critic, label=f'critic (x{scale_critic})')
        plt.plot(np.array(data['reward'])/scale_reward, label=f'avg.reward (x{scale_reward})')
        plt.plot(np.array(data['traverse'])/scale_meters, label=f'avg.traverse', alpha=0.4)
        plt.plot(np.abs(np.array(data['side']))/scale_meters, label=f'avg.abs.side', alpha=0.4)
        #plt.plot(np.array(data['pitch'])/180, label=f'avg.pitch (deg/180deg)', alpha=0.4, ls="--")
        #plt.plot(np.array(data['reward_sum']), label=f'rew_sum', alpha=0.4)
        plt.title("Actor/Critic Loss (" + str(data['epsiode'][-1]) + "/" + str(data['num_learning_iterations'][0]) + ")")
        plt.ylabel("Loss")
        plt.xlabel("Episodes")
        plt.ylim([-1, 4]) # lock y-axis
        #plt.grid(True)
        plt.legend()
        if savefig:
            plt.savefig(os.path.join(save_dir, 'ac_loss.png'))
        if running_plot:
            plt.draw()
            plt.pause(0.1)
        else:
            plt.show()

    def save_model(self, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir,filename))
        torch.save(self.state_dict(), os.path.join(save_dir, f"model/model.pt"))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def save_data(save_dir: str, data_dict: defaultdict, filename: str = "data"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = open(os.path.join(save_dir, filename + ".pkl"), "wb")
        pickle.dump(data_dict, f)
        f.close()

