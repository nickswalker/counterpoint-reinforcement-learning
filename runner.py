import copy
import sys
from typing import Tuple, List

import numpy as np
import scipy as scipy
import scipy.stats
from abjad.tools import lilypondfiletools
from abjad.tools.indicatortools.KeySignature import KeySignature
from abjad.tools.metertools.Meter import Meter
from abjad.tools.pitchtools.NamedPitch import NamedPitch
from abjad.tools.scoretools import Voice

from cantus_firmi import cantus_firmi
from counterpoint.composition_environment import CompositionEnvironment
from counterpoint.species_counterpoint import SpeciesOneCounterpoint
from rl.agent.qlearning import QLearning
from rl.agent.sarsa import Sarsa
from rl.agent.trueonlinesarsalambda import TrueOnlineSarsaLambda
from rl.domain import Domain
from rl.task import Task

evaluation_period = 50
significance_level = 0.05

stochasticity = 0.0


def main():
    experiment_num = int(sys.argv[1])
    num_evaluations = int(sys.argv[2])
    num_trials = int(sys.argv[3])

    def save(name, results):
        data = np.c_[results]
        np.savetxt("results/" + str(experiment_num) + "_" + str(num_trials) + "_" + name + ".csv", data,
                   fmt=["%d", "%f", "%f", "%f"],
                   delimiter=",")

    if experiment_num == 0:
        environment_factory = make_environment_factory([cantus_firmi[0]])
        agent_factory = make_agent_factory(q_learning=True)
        q_learning_results = run_experiment(num_trials, num_evaluations, agent_factory, environment_factory)
        save("Q-learning", q_learning_results)


def run_experiment(num_trials, num_evaluations,
                   agent_factory, environment_factory
                   ):
    assert num_trials > 1
    evaluations_mean = []
    evaluations_variance = []
    series = [i * evaluation_period for i in range(0, num_evaluations)]
    n = 0
    for i in range(0, num_trials):
        print("trial " + str(i))
        j = int(0)
        n += 1
        for (num_episodes, table) in train_agent(evaluation_period,
                                                 num_evaluations,
                                                 agent_factory,
                                                 environment_factory):
            evaluation = evaluate(table, agent_factory, environment_factory)
            # print(" R: " + str(evaluation))
            mean = None
            variance = None
            if j > len(evaluations_mean) - 1:
                evaluations_mean.append(0.0)
                evaluations_variance.append(0.0)
                mean = 0.0
                variance = 0.0
            else:
                mean = evaluations_mean[j]
                variance = evaluations_variance[j]

            delta = evaluation - mean
            mean += delta / n
            variance += delta * (evaluation - mean)

            evaluations_mean[j] = mean
            evaluations_variance[j] = variance
            j += 1

    evaluations_variance = [variance / (n - 1) for variance in
                            evaluations_variance]

    confidences = []
    for (mean, variance) in zip(evaluations_mean, evaluations_variance):
        crit = scipy.stats.t.ppf(1.0 - significance_level, n - 1)
        width = crit * np.math.sqrt(variance) / np.math.sqrt(n)
        confidences.append(width)

    return series, evaluations_mean, evaluations_variance, confidences


def evaluate(table, agent_factory, environment_factory) -> float:
    domain, task = environment_factory()
    agent = agent_factory(domain, task)
    agent.value_function = table
    agent.epsilon = 0.0
    agent.alpha = 0.0
    cumulative_reward = 0.0
    terminated = False
    max_steps = 200
    current_step = 0

    trajectory = []

    while not terminated:
        current_step += 1
        agent.act()

        trajectory.append((agent.previousstate, agent.previousaction))

        if task.stateisfinal(domain.get_current_state()) or current_step > max_steps:
            terminated = True
            cumulative_reward = agent.get_cumulative_reward()
            agent.episode_ended()

    save_composition("eval", domain)
    return cumulative_reward


def train_agent(evaluation_period, num_stops, agent_factory, environment_factory):
    """
    Trains an agent, periodically yielding the agent's q-table
    :param evaluation_period:
    :param num_stops:
    :param initial_value:
    :param epsilon:
    :param alpha:
    :return:
    """
    domain, task = environment_factory()
    agent = agent_factory(domain, task)

    stops = 0
    for i in range(0, evaluation_period * num_stops):
        if i % evaluation_period is 0:
            # print(i)
            stops += 1
            yield i, copy.deepcopy(agent.value_function)

        if num_stops == stops:
            return
        terminated = False
        max_steps = 200
        current_step = 0
        while not terminated:
            current_step += 1
            agent.act()

            if task.stateisfinal(domain.get_current_state()):
                final_state = domain.get_current_state()
                agent.episode_ended()
                domain.reset()
                terminated = True


def make_environment_factory(given_voices: List[Voice], meter=Meter(4, 4), key=KeySignature('c', 'major')):
    def generate_environment() -> Tuple[Domain, Task]:
        domain = CompositionEnvironment(2, given_voices, meter, key)
        task = SpeciesOneCounterpoint(domain,
                                      [(NamedPitch("C3"), NamedPitch("C5")), (NamedPitch("C2"), NamedPitch("C4"))])
        return domain, task

    return generate_environment


def make_agent_factory(initial_value=0.5,
                       epsilon=0.1,
                       alpha=0.2,
                       lmbda=0.95,
                       expected=False, true_online=False, q_learning=False):
    def generate_agent(domain, task):
        if true_online:
            agent = TrueOnlineSarsaLambda(domain, task, epsilon=epsilon, alpha=alpha, lamb=lmbda, expected=False)
        elif q_learning:
            agent = QLearning(domain, task)
        else:
            agent = Sarsa(domain, task, epsilon=epsilon, alpha=alpha, expected=expected)
        return agent

    return generate_agent


def save_composition(name: str, composition: CompositionEnvironment):
    lilypond_file = lilypondfiletools.make_basic_lilypond_file(composition.given_voices[0])
    filename = name + ".ly"
    with open("results/" + filename, mode="w") as f:
        f.write(format(lilypond_file))

if __name__ == '__main__':
    main()
