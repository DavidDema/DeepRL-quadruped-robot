import os.path
from env.go_env import GOEnv
from rl_modules.storage import Storage
import torch
import torch.nn as nn
import torch.optim as optim
from rl_modules.actor_critic import ActorCritic

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
                 action_scale=0.3
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

    def act(self, obs):
        """
        Sample an action (and according value and log-prob) from given policy (parameter obs)

        :param obs: observation (concat[qpos, qvel])
        :return: action
        """

        action = self.actor_critic.act(obs).squeeze()
        self.transition.action = action.detach().cpu().numpy()
        self.transition.value = self.actor_critic.evaluate(obs).squeeze().detach().cpu().numpy()
        self.transition.action_log_prob = self.actor_critic.get_actions_log_prob(action).detach().cpu().numpy()
        return self.transition.action

    def inference(self, obs):
        """
        Sample an action from given policy (parameter obs)

        :param obs: observation (concat[qpos, qvel])
        :return: action
        """
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

    def update(self):
        """ Update"""

        mean_value_loss = 0
        mean_actor_loss = 0

        generator = self.storage.mini_batch_generator(self.num_batches, self.num_epochs, device=self.device)

        # For batches (default=1) and epochs (default=1) TODO: What are the batches and epochs ?
        for obs_batch, actions_batch, target_values_batch, advantages_batch in generator:
            # Evaluate policy
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            # Compute losses
            actor_loss = (-advantages_batch * actions_log_prob_batch).mean()
            critic_loss = advantages_batch.pow(2).mean()
            loss = actor_loss + self.value_loss_coef * critic_loss

            # Gradient step - Update parameters
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
        """
        Play for one Episode
        :param is_training: Training mode enabled if True
        :param early_termination: Early termination enabled if True
        :return: Information
        """

        # Reset environment and get current position
        obs, _ = self.env.reset()
        infos = []
        i = 0
        # Start one episode (can also be multiple attempts when the robot is terminated, TODO maybe change that ?)
        for _ in range(self.storage.max_timesteps):
            i += 1
            if i%100==0:
                print(i)
            obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(dim=0)
            with torch.no_grad():
                # Select action
                if is_training:
                    # Sample a random action from updated distribution
                    # (updated using the action_mean=actorNN(obs_tensor))
                    action = self.act(obs_tensor)
                else:
                    # Sample the action using the actor -> action_mean=actorNN(obs_tensor)
                    action = self.inference(obs_tensor)

            # Perform one step (simulate, calculate rewards and check early termination)
            obs_next, reward, terminate, info = self.env.step(action*self.action_scale)
            infos.append(info)
            if is_training:
                # Collect data for storage
                self.store_data(obs, reward, terminate)
            if terminate and early_termination:
                # Reset env if player is in bad situation
                # Keep episode alive (TODO:why?)
                obs, _ = self.env.reset()
            else:
                # Update current position
                obs = obs_next
        if is_training:
            # Compute Q(s,a) for this episode (from T->0) using the Advantage function
            self.compute_returns(torch.from_numpy(obs_next).to(self.device).float())

        return infos

    def learn(self, save_dir, num_learning_iterations=1000, num_steps_per_val=50):
        # For each learning iteration, play one episode
        for it in range(num_learning_iterations):
            print(it)
            # play games to collect data - play one episode
            infos = self.play(is_training=True)
            # improve policy with collected data - from one episode
            mean_value_loss, mean_actor_loss = self.update()

            # Validate model after <num_steps_per_val> iterations and save model
            if it % num_steps_per_val == 0:
                infos = self.play(is_training=False)
                self.save_model(os.path.join(save_dir, f'{it}.pt'))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))



