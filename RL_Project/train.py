from rl_modules.rl_agent import RLAgent
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import wandb
from datetime import datetime
import os
import torch

from utils_cheetah.config import Config

def train():
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    C = Config(dir="config", filename="config.yaml")
    cfg = C.load()

    # create environment
    if cfg['runner']['render_human']:
        go_env = GOEnv(render_mode="human", cfg=cfg)
    else:
        go_env = GOEnv(cfg=cfg)  # GOEnv(render_mode="human")

    # create actor critic
    actor_critic = ActorCritic(state_dim=go_env.obs_dim, action_dim=go_env.action_dim, cfg=cfg).to(device)
    # create storage to save data
    storage = Storage(obs_dim=go_env.obs_dim, action_dim=go_env.action_dim, cfg=cfg)
    rl_agent = RLAgent(env=go_env, actor_critic=actor_critic, storage=storage, device=device, cfg=cfg)

    save_dir = f'checkpoints/{log_name}'
    save_dir_model = f'{save_dir}/model'
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)
    C.save(save_dir_model)

    # Learn
    rl_agent.learn(save_dir)

if __name__ == '__main__':
    train()
