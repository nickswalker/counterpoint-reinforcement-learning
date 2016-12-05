import argparse
import os

import numpy as np

from cantus_firmi import cantus_firmi
from counterpoint.species_counterpoint import SpeciesOneCounterpoint
from counterpoint.test_tasks import ScalesAreGood
from utilities.factories import make_environment_factory, make_agent_factory, Approach
from utilities.save_composition import save_composition
from utilities.trial_log import ExperimentLog

significance_level = 0.05


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=int)
    parser.add_argument("outdir", type=str)
    parser.add_argument('-trials', type=int, default=1)
    parser.add_argument('-evaluations', type=int, default=10)
    parser.add_argument('-period', type=int, default=100)
    parser.add_argument('-unique-id', type=int)
    parser.add_argument('--log-evaluations', type=int)
    parser.add_argument("-history", type=int, default=3)
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--no-feature', dest='feature', )

    parser.add_argument("--time-invariant-state", action='store_true', default=False)
    args = parser.parse_args()

    experiment_num = args.experiment
    num_trials = args.trials
    num_evaluations = args.evaluations
    evaluation_period = args.period
    output_dir = args.outdir
    history_length = args.history
    time_invariant_state = args.time_invariant_state

    meter = cantus_firmi[0][1]
    key = cantus_firmi[0][2]
    results = None
    if experiment_num == 1:
        environment_factory = make_environment_factory([], meter, key, ScalesAreGood, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.QLearning)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "Q-learning"
    elif experiment_num == 2:
        environment_factory = make_environment_factory([], meter, key, SpeciesOneCounterpoint, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.QLearning, epsilon=0.4)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "Q-learning"

    elif experiment_num == 3:
        environment_factory = make_environment_factory([], meter, key, ScalesAreGood, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.QNetwork, alpha=0.01)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "Q-network"
    elif experiment_num == 4:
        environment_factory = make_environment_factory([], meter, key, SpeciesOneCounterpoint, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.QNetwork)
        qnetwork_results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                          environment_factory, output_dir)
        agent_name = "Q-network"
    elif experiment_num == 5:
        environment_factory = make_environment_factory([], meter, key, SpeciesOneCounterpoint, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.DDDQN, alpha=0.001, epsilon=1.0)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "Double Dueling DQN"
    elif experiment_num == 6:
        environment_factory = make_environment_factory([], meter, key, ScalesAreGood, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.DDDQN, alpha=0.5, epsilon=1.0)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "Double Dueling DQN"
    elif experiment_num == 7:
        environment_factory = make_environment_factory([], meter, key, SpeciesOneCounterpoint, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.Sarsa, epsilon=0.4, alpha=0.1, lmbda=0.5)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "True Online Sarsa(l)"
    elif experiment_num == 8:
        environment_factory = make_environment_factory([], meter, key, SpeciesOneCounterpoint, history_length,
                                                       time_invariant_state)
        agent_factory = make_agent_factory(Approach.TrueOnlineSarsaLambda, epsilon=0.4)
        results = run_experiment(num_trials, num_evaluations, evaluation_period, agent_factory,
                                 environment_factory, output_dir)
        agent_name = "True Online Sarsa(l)"

    save(agent_name, experiment_num, results, output_dir, args.unique_id)


def run_experiment(num_trials, num_evaluations, evaluation_period,
                   agent_factory, environment_factory, out_dir
                   ) -> ExperimentLog:
    series = [i * evaluation_period for i in range(0, num_evaluations)]
    log = ExperimentLog(series, 0.05)
    evaluation_num = 0
    for i in range(0, num_trials):
        print("trial " + str(i))

        # Train and periodically yield the value function
        for (num_episodes, table) in train_agent(evaluation_period,
                                                 num_evaluations,
                                                 agent_factory,
                                                 environment_factory):
            evaluation = evaluate(table, agent_factory, environment_factory, "Evaluation %d" % evaluation_num, out_dir)
            evaluation_num += 1
            print(" R: " + str(evaluation))
            log.observe(evaluation)
        log.observe_trial_end()

    return log


def evaluate(table, agent_factory, environment_factory, unique_name: str, out_dir: str) -> float:
    domain, task = environment_factory()
    agent = agent_factory(domain, task, table)
    agent.value_function = table
    agent.epsilon = 0.0
    agent.alpha = 0.0
    cumulative_reward = 0.0
    terminated = False
    current_step = 0

    while not terminated:
        current_step += 1
        agent.act()

        if task.stateisfinal(domain.get_current_state()):
            terminated = True
            cumulative_reward = agent.get_cumulative_reward()
            agent.episode_ended()

    save_composition(unique_name, agent.name, domain, out_dir)
    return cumulative_reward


def save(name: str, experiment_num: int, log: ExperimentLog, out_dir: str, unique_num: int = 0):
    out_prefix = out_dir

    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
    filename = str(experiment_num) + "_" + str(log.n) + "_" + name + str(unique_num) + ".csv"
    full_out_path = os.path.join(out_prefix, filename)
    if log.n > 1:
        log.finalize_confidences()
        data = np.c_[(log.series, log.means, log.variances, log.confidences)]
        formats = ["%d", "%f", "%f", "%f"]
    else:
        data = np.c_[(log.series, log.means)]
        formats = ["%d", "%f"]
    np.savetxt(full_out_path, data,
               fmt=formats,
               delimiter=",")


def train_agent(evaluation_period, num_stops, agent_factory, environment_factory):
    """
    Trains an agent, periodically yielding the agent's q-table
    :param evaluation_period:
    :param num_stops:

    :return:
    """
    domain, task = environment_factory()
    agent = agent_factory(domain, task, None)
    reward = 0
    stops = 0
    for i in range(0, evaluation_period * num_stops):
        if i % evaluation_period is 0:
            stops += 1
            yield i, agent.value_function

        if num_stops == stops:
            return
        terminated = False

        current_step = 0
        while not terminated:
            current_step += 1
            agent.act()

            if task.stateisfinal(domain.get_current_state()):
                final_state = domain.get_current_state()
                reward = agent.get_cumulative_reward()
                agent.episode_ended()
                domain.reset()
                terminated = True

        if i % 10 == 0:
            print("%d: %f" % (i, reward))


if __name__ == '__main__':
    main()
