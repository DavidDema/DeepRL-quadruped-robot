import pickle

import numpy as np
from matplotlib import pyplot as plt

from rl_modules.rl_agent import RLAgent
import os

from config.logger import load_experiment

def main():
    exp_names = ["r1", "r2"]
    datas = []
    for name in exp_names:
        data, cfg = load_exp(name)
        datas.append(data)
    config = cfg
    plot_variance(datas, config)

def load_exp(name):
    experiment = name
    dir = f'checkpoints/{experiment}'
    if not os.path.exists(dir):
        print("Directory does not exist !")
        return False

    data, cfg = load_experiment(dir)
    return data, cfg

def plot_variance(datas, cfg):
    it = 0
    its = []
    for data in datas:
        its.append(np.max(np.array(data['epsiode'])))
    it = np.min(np.array(its))
    epsiodes = datas[0]['epsiode'][:it]

    rewards = []
    for data in datas:
        rewards.append(np.array(data['reward'])[:it])
    rew = np.reshape(np.array(rewards), (len(datas), it))
    mean_rew = np.mean(rew, axis=0)
    var_rew = np.var(rew, axis=0)

    plt.figure(1)
    plt.plot(epsiodes, mean_rew, lw=2, label="mean_rew")
    plt.fill_between(epsiodes, mean_rew-var_rew, mean_rew+var_rew, alpha=0.2, lw=1.5)
    #plt.fill_between(epsiodes, mean_rew, 1)
    #plt.plot(var_rew)
    plt.plot(rew[0], alpha=0.4, label="rew1")
    plt.plot(rew[1], alpha=0.4, label="rew2")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()
