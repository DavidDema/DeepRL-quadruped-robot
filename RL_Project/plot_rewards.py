import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from rl_modules.rl_agent import RLAgent
import os

from config.logger import load_experiment

def main():
    exp_names = ["r15", "r8", "r9", "r13", "r10", "r11", "r12", "r14"]
    datas = []
    for name in exp_names:
        data, cfg = load_exp(name)
        datas.append(data)
    config = cfg
    plot_rewards(datas, config, exp_names)

def load_exp(name):
    dir = f'results/{name}'
    if not os.path.exists(dir):
        print("Directory does not exist !")
        return False

    data, cfg = load_experiment(dir)
    return data, cfg

def plot_rewards(datas, cfg, exp_names):
    datadict = {
        'r15': 'our method',
        'r8':  'w/o GAE/PPO',
        'r9':  'w/o GAE',
        'r13': 'w/o PPO',
        'r10': 'w/ Dropout',
        'r11': 'max_timesteps=500',
        'r12': 'max_timesteps=1000',
        'r14': 'w/ LSTM'
    }

    keys = []
    for key in datas[0].keys():
        if key.startswith("reward_") and np.sum(datas[0][key]) and key != "reward_sum":
            keys.append(key)
    
    num_plots = len(keys)
    num_rows = 3  # Change this based on the number of rows you want
    num_cols = (num_plots + num_rows - 1) // num_rows

    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    for i, key in enumerate(keys, 1):
        ax = plt.subplot(num_rows, num_cols, i)

        for j, data in enumerate(datas):
            data2 = data[key]
            x_smooth = np.linspace(0, len(data2) - 1, 20)
            spl = make_interp_spline(range(len(data2)), data2, k=3)
            y_smooth = spl(x_smooth)
            linewidth, alpha = 1, 0.7
            if datadict[exp_names[j]] == "our method":
                linewidth, alpha = 2, 0.8
            plt.plot(x_smooth, y_smooth, alpha=alpha, linestyle='solid', linewidth=linewidth, label=datadict[exp_names[j]])

        if i % num_cols == 1:  # For leftmost subplots
            plt.ylabel('Rewards')

        if i > (num_rows - 1) * num_cols:  # For bottom subplots
            plt.xlabel('Episode')

        plt.title(key[len("reward_"):])

    # Create a common legend outside the loop
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout(h_pad=0.3, w_pad=0.3)
    plt.show()

if __name__ == '__main__':
    main()
