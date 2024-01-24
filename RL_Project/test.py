import os

from rl_modules.rl_agent import RLAgent
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import torch

from deeprl_cheetah.RL_Project.config.config import Config

def test():
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'

    # Select the training session path)
    use_experiment = True
    if use_experiment:
        # Use the checkpoint/experiment folder
        save_dir = 'checkpoints/'
        experiment = 'latest'
        filepath_model = os.path.join(save_dir, f"{experiment}/model.pt")
        filepath_cfg = os.path.join(save_dir, f"{experiment}/config.yaml")
    else:
        # Use the checkpoint/.../ folder
        year = "2024"
        month = "01"
        day = "24"
        hour = "07"
        minute = "43"
        second = "37"
        iteration_nr = 700  # select iteration number
        log_name = os.path.join((year + "-" + month + "-" + day), (hour + "-" + minute + "-" + second))
        save_dir = f'checkpoints/{log_name}'
        if not os.path.exists(save_dir):
            print("Directory does not exist !")
            return False
        filepath_model = os.path.join(save_dir, f"{iteration_nr}.pt")
        filepath_cfg = os.path.join(save_dir, f"model/config.pt")

    # Load config file
    c = Config(path=filepath_cfg)
    cfg = c.load()

    # create environment
    go_env = GOEnv(render_mode="human", cfg=cfg)
    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim, cfg=cfg).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, cfg=cfg)#, max_timesteps=2000)
    # remove cfg for different max_timesteps
    rl_agent = RLAgent(env=go_env, actor_critic=actor_critic, storage=storage, device=device, cfg=cfg)

    # load parameter model
    rl_agent.load_model(filepath_model)
    #rl_agent.load_model('checkpoints/latest/model.pt')

    # start episode
    rl_agent.play(is_training=False)


if __name__ == '__main__':
    test()
