import pickle

import numpy as np
from matplotlib import pyplot as plt

from rl_modules.rl_agent import RLAgent
import os

from config.logger import load_experiment, load_data

def main():
    exp_names = ["r3", "r4", "r7"]
    #exp_names = ["r3", "r5", "r6"]
    datas = []
    for name in exp_names:
        data = load_data("results/" + name + "/test_data.pkl")
        datas.append(data)
    plot_testrun(datas, exp_names)


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

def plot_testrun(datas, names):

    traverses = load_datas(datas, key="traverse")
    side = load_datas(datas, key="side")
    speedx = load_datas(datas, key="speedx", smooth=True, kernel_size=20)
    terminates = load_datas(datas, key="terminate")

    plt.figure(1)
    for j in range(len(traverses)):
        travs, sides = split_traverse(traverses[j], side[j], terminates[j])
        for i in range(len(travs)):
            plt.plot(travs[i], sides[i], ls="-", lw=0.8, alpha=1.0, label=f"Runner {j+1}")
            plt.plot(travs[i][-1], sides[i][-1], marker="x", lw=1.2, color="red")
    plt.legend()
    plt.ylim([-0.5, 0.5])
    plt.xlabel("Longitude [m]")
    plt.ylabel("Transversal [m]")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/test_traverse.png")
    plt.show()
    plt.close()

    plt.figure(1)
    for i in range(len(speedx)):
        plt.plot(speedx[i], ls="-", lw=1.0, alpha=0.8, label=f"Runner {i + 1}")
    plt.axhline(y=0.5, ls="--", lw=0.8, c="red", label="Target velocity")
    plt.legend()
    plt.ylabel("Velocity of robot base [m/s]")
    plt.xlabel("Simulation time [s]")
    plt.ylim([None, 0.55])
    plt.grid(True, alpha=0.3)
    plt.savefig("results/test_speed.png")
    plt.show()


if __name__ == '__main__':
    main()
