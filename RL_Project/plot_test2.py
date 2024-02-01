import pickle

import numpy as np
from matplotlib import pyplot as plt

from rl_modules.rl_agent import RLAgent
import os

from config.logger import load_experiment, load_data

def main():
    exp_names = ["r3", "r7", "r14"]
    #exp_names = ["r3", "r5", "r6"]
    label = ["Experiment 1", "Experiment 2", "Experiment 3 w/ LSTM"]

    datas = []

    test_runs = 4
    for name in exp_names:
        datas2 = []
        for i in range(test_runs):
            datas2.append(load_data("results/" + name + f"/test_data{i}.pkl"))
        datas.append(datas2)

    plot_testrun(datas, exp_names, label)


def load_datas(datas, key, it=1000, smooth=False, kernel_size=10):

    if not it:
        # find lowest iteration
        its = []
        for data in datas:
            its.append(np.max(np.array(data['epsiode'])))
        it = np.min(np.array(its))

    values = []
    for data in datas:
        values.append(np.array(data[key])[:it])
    val = np.reshape(np.array(values), (len(datas), it))
    if smooth:
        for i in range(len(val)):
            val[i] = smooth_data(val[i], kernel_size)
    return val


def smooth_data(values, kernel_size=10):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(values, kernel, mode='same')

def split_traverse(traverse, side, terminate):
    travs = []
    sides = []

    t = []
    s = []
    for i, term in enumerate(terminate):
        t.append(traverse[i])
        s.append(side[i])

        if term or i==len(terminate)-1:
            travs.append(t)
            sides.append(s)
            t = []
            s = []
    return travs, sides

def plot_testrun(datas2, names, labels):

    traverses = []
    sides = []
    speedxs = []
    terminates = []
    for i in range(len(datas2)):
        datas = datas2[i]
        traverses.append(load_datas(datas, key="traverse"))
        sides.append(load_datas(datas, key="side"))
        speedxs.append(load_datas(datas, key="speedx", smooth=True, kernel_size=20))
        terminates.append(load_datas(datas, key="terminate"))

    plt.figure(1)
    for i_exp in range(len(datas2)):
        traverse = traverses[i_exp]
        side_i = sides[i_exp]
        for j in range(len(traverse)):
            travs = traverse[j]
            side = side_i[j]
            if j == 0:
                plt.plot(travs, side, ls="-", c=f"C{i_exp}", lw=0.8, alpha=1.0, label=f"{labels[i_exp]}")
            else:
                plt.plot(travs, side, ls="-", c=f"C{i_exp}", lw=0.8, alpha=1.0)
            plt.plot(travs[-1], side[-1], marker="x", lw=1.2, color="red")
    plt.legend()
    #plt.ylim([-0.5, 0.5])
    plt.xlabel("Position in x [m]")
    plt.ylabel("Position in y [m]")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/test_traverse.png")
    plt.show()
    plt.close()


    plt.figure(1)
    for i_exp in range(len(datas2)):
        speedx = speedxs[i_exp]
        mean_speedx = np.mean(speedx, axis=0)
        plt.plot(mean_speedx, ls="-", c=f"C{i_exp}", lw=1.2, alpha=0.8, label=f"{labels[i_exp]}")

    plt.axhline(y=0.5, ls="--", lw=0.8, c="red", label="Target velocity")
    plt.legend()
    plt.ylabel("Velocity of robot base [m/s]")
    plt.xlabel("Simulation time [s]")
    plt.ylim([0, 0.6])
    plt.grid(True, alpha=0.3)
    plt.savefig("results/test_speed.png")
    plt.show()


if __name__ == '__main__':
    main()
