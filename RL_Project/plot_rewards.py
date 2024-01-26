import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from rl_modules.rl_agent import RLAgent
import os

from config.logger import load_experiment

def main():
    exp_names = ["r8", "r8", "r8", "r8", "r9", "r10"]
    datas = []
    for name in exp_names:
        data, cfg = load_exp(name)
        datas.append(data)
    config = cfg
    plot_rewards(datas, config)

def load_exp(name):
    dir = f'results/{name}'
    if not os.path.exists(dir):
        print("Directory does not exist !")
        return False

    data, cfg = load_experiment(dir)
    return data, cfg

def plot_rewards(datas, cfg):
    num_plots = len(datas)
    num_rows = 2  # Change this based on the number of rows you want
    num_cols = (num_plots + num_rows - 1) // num_rows

    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    for i, data in enumerate(datas, 1):
        plt.subplot(num_rows, num_cols, i)
        
        for key in data.keys():
            if key.startswith("reward_") and np.sum(data[key]) and key != "reward_sum":
                data2 = data[key]
                x_smooth = np.linspace(0, len(data2) - 1, 40)
                spl = make_interp_spline(range(len(data2)), data2, k=3)
                y_smooth = spl(x_smooth)
                plt.plot(x_smooth, y_smooth, alpha=0.8, linestyle='solid', linewidth=1, label=key[len("reward_"):])

        plt.legend()
        plt.title(f'Plot {i}')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()

if __name__ == '__main__':
    main()
