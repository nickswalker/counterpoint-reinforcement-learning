import os
import sys
from typing import List

import numpy as np
import scipy as sp
import scipy.stats

from utilities.plotting import Plot


def main():
    figure_num = int(sys.argv[1])
    for_print = bool(int(sys.argv[2]))

    def load_and_plot(dir: str, plot: Plot, name: str):
        series, means, confidences = load(dir)
        plot.plot_evaluations(series, means, confidences, name)

    if figure_num == 0:
        plot = Plot("Mean evaluation grade", for_print, small=True)
        load_and_plot("results/s0-q-1-1k/collected", plot, "k=1")
        plot.save("figure0", "report")
    elif figure_num == 1:
        plot = Plot("Mean evaluation grade", for_print, small=True)
        load_and_plot("results/cmac-1-Wed-Dec-07-02-38/collected", plot, "k=1")
        load_and_plot("results/cmac-2-Wed-Dec-07-02-37/collected ", plot, "k=2")
        load_and_plot("results/cmac-3-Wed-Dec-07-02-37/collected", plot, "k=3")
        plot.save("figure1", "report")
    elif figure_num == 2:
        plot = Plot("Mean evaluation grade", for_print, small=True)
        load_and_plot("results/cmac-1-inv-Wed-Dec-07-02-38/collected", plot, "k=1")
        load_and_plot("results/cmac-2-inv-Wed-Dec-07-02-38/collected ", plot, "k=2")
        plot.save("figure2", "report")




def load(dir: str):
    trials = get_trials(dir)
    rewards_by_step = extract_data(trials)

    means = []
    confidences = []
    for rewards in rewards_by_step:
        mean, confidence = mean_confidence_interval(rewards)
        means.append(mean)
        confidences.append(confidence)

    series = [i * 100 for i in range(0, len(means))]

    return series, means, confidences


def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h * 2


def extract_data(paths: List[str]) -> List[List[float]]:
    rewards_by_episode = [[] for i in range(0, 1000)]
    for path in paths:
        episodes, rewards, _, _ = np.loadtxt(path, delimiter=",").T
        i = 0
        for (steps, reward) in zip(episodes, rewards):
            rewards_by_episode[i].append(reward)
            i += 1

    rewards_by_episode = [episode for episode in rewards_by_episode if len(episode) > 0]
    return rewards_by_episode[0:min(200, len(rewards_by_episode))]


def get_trials(dir: str) -> List[str]:
    dir = dir.strip()
    return [os.path.join(dir, name) for name in os.listdir(dir) if
            os.path.isfile(os.path.join(dir, name)) and not name.startswith(".") and name.endswith(".csv")]


main()
