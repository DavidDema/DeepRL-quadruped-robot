import shutil

from rl_modules.rl_agent import RLAgent
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
from datetime import datetime
import os
import torch

from config.logger import load_cfg, save_cfg, load_data, load_experiment

def train():
    if torch.cuda.is_available():
        device ='cuda'
    else:
        device = 'cpu'
    log_name = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    cfg = load_cfg("config/config.yaml")

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

    # save the cfg-file in the checkpoint folder checkpoints/.../model/
    save_dir = f'checkpoints/{log_name}'
    save_dir_model = os.path.join(save_dir, "model")
    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)
    save_cfg(cfg=cfg, save_dir=save_dir_model)
    # remove the folder for the latest results
    try:
        shutil.rmtree("checkpoints/latest", )
    except:
        pass

    # learn
    rl_agent.learn(save_dir)


if __name__ == '__main__':
    train()
