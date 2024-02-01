import pickle

import numpy as np
from matplotlib import pyplot as plt

from rl_modules.rl_agent import RLAgent
import os

from config.logger import load_experiment

def main():
    exp_name = "r3"

    data, cfg = load_exp(exp_name)
    config = cfg
    plot_variance(data, config)

def load_exp(name):
    experiment = name
    dir = f'results/{experiment}'
    if not os.path.exists(dir):
        print("Directory does not exist !")
        return False

    data, cfg = load_experiment(dir)
    return data, cfg


def load_data(data, key, scale: float = 1.0, start_iter=0, kernel_size=50, it=None):

    if not it:
        it = np.max(np.array(data['epsiode']))

    episodes = data['epsiode'][start_iter:it+10]

    val = np.array(data[key])[start_iter:it+10]
    mean_val = smooth_data(val, kernel_size=kernel_size)
    var_val = np.zeros(np.shape(mean_val))

    return val/scale, mean_val/scale, var_val/scale, episodes, it


def smooth_data(values, kernel_size=30):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(values, kernel, mode='same')


def mean_var(values):
    mean = smooth_data(values)
    var1 = np.zeros(len(mean))
    var2 = np.zeros(len(mean))

    vars = values-mean
    for t in range(len(mean)):
        if vars[t] < 0:
            var1[t] = vars[t]
        else:
            var2[t] = vars[t]
    return mean, var1, var2


def plot_variance(data, cfg):

    scale_ac = 0.01
    scale_cr = 10000
    scale_rew = 10

    iteration = 1500

    rewards, rewards_mean, rewards_var, episodes, it = load_data(data, key="reward", scale=scale_rew, kernel_size=50, it=iteration)
    cr_losses, cr_losses_mean, cr_losses_var, episodes, it = load_data(data, key="critic_loss", scale=scale_cr, kernel_size=50, it=iteration)
    ac_losses, ac_losses_mean, ac_losses_var, episodes, it = load_data(data, key="actor_loss", scale=scale_ac, kernel_size=50, it=iteration)

    labelsize1 = 11
    labelsize2 = 8

    plt.figure(1, dpi=300)

    i1 = 0
    i2 = 1400
    plt.plot(episodes[i1:i2], ac_losses_mean[i1:i2], c="C0", lw=1.5, label=f"mean actor loss (x{scale_ac})")
    plt.plot(episodes[i1:i2], ac_losses[i1:i2], c="C0", alpha=0.2)

    plt.plot(episodes[i1:i2], cr_losses_mean[i1:i2], c="C1", lw=1.5, label=f"mean critic loss (x{scale_cr})")
    plt.plot(episodes[i1:i2], cr_losses[i1:i2], c="C1", alpha=0.2)

    plt.plot(episodes[i1:i2], rewards_mean[i1:i2], c="C2", lw=1.5, label=f"avg. total reward (x{scale_rew})")
    plt.plot(episodes[i1:i2], rewards[i1:i2], c="C2", alpha=0.2)

    plt.axhline(y=0, ls="--", alpha=0.4, color="grey")
    plt.legend(loc="lower right", fontsize=labelsize1)
    plt.ylim([-2.5, 3])
    plt.xlabel("Episodes [1]", fontsize=labelsize1)

    plt.savefig("results/best_results_r3.png")
    plt.show()


if __name__ == '__main__':
    main()
