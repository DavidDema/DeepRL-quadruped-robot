import os

from rl_modules.rl_agent import RLAgent
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import torch

from utils_cheetah.config import Config

def test():
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'

    use_default = False
    if use_default:
        save_dir = 'checkpoints/'
        filepath_model = os.path.join(save_dir, f"model/model.pt")
    else:
        year = "2024"
        month = "01"
        day = "24"
        hour = "07"
        minute = "43"
        second = "37"
        iteration_nr = 700
        log_name = os.path.join((year + "-" + month + "-" + day), (hour + "-" + minute + "-" + second))
        save_dir = f'checkpoints/{log_name}'
        if not os.path.exists(save_dir):
            print("Directory does not exist !")
            return False
        filepath_model = os.path.join(save_dir, f"{iteration_nr}.pt")

    C = Config(dir=os.path.join(save_dir, "model"), filename="config.yaml")
    cfg = C.load()

    # create environment
    go_env = GOEnv(render_mode="human", cfg=cfg)
    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim, cfg=cfg).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, cfg=cfg)#, max_timesteps=2000)
    # remove cfg for different max_timesteps
    rl_agent = RLAgent(env=go_env, actor_critic=actor_critic, storage=storage, device=device, cfg=cfg)

    rl_agent.load_model(filepath_model)
    #rl_agent.load_model('checkpoints/model/model.pt')
    rl_agent.play(is_training=False)


if __name__ == '__main__':
    test()
