import pickle

from rl_modules.rl_agent import RLAgent
from env.go_env import GOEnv
from rl_modules.storage import Storage
from rl_modules.actor_critic import ActorCritic
import wandb
from datetime import datetime
import os
import torch

from utils_cheetah.config import Config

def plot():
    year = "2024"
    month = "01"
    day = "23"
    hour = "09"
    minute = "50"
    second = "57"
    log_name = os.path.join((year + "-" + month + "-" + day),(hour + "-" + minute + "-" + second))

    save_dir = f'checkpoints/{log_name}'
    if not os.path.exists(save_dir):
        print("Directory does not exist !")
        return False

    C = Config(dir=save_dir, filename="config.yaml")
    cfg = C.load()

    with open(os.path.join(save_dir, 'data.pkl'), 'rb') as file:
        data = pickle.load(file)

    #data['num_learning_iterations'].append(500)
    RLAgent.plot_results(save_dir, data)


if __name__ == '__main__':
    plot()
