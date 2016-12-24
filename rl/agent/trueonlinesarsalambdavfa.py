import random

import numpy as np

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task
from rl.valuefunction import FeatureExtractor
from rl.valuefunction.linear import LinearVFA


class TrueOnlineSarsaLambdaVFA(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.6,
                 gamma=1.0, lamb=0.95, name="True Online Sarsa(λ)"):
        """
        :param domain: The world the agent is placed in.
        :param task: The task in the world, which defines the reward function.
        """
        super().__init__(domain, task, name)
        self.world = domain
        self.task = task
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.previousaction = None
        self.previousstate = None
        self.value_old = 0.0

        actions = domain.get_actions()
        self.feature_extractor = feature_extractor
        self.eligibility = np.zeros(self.feature_extractor.num_features())
        self.value_function = LinearVFA(self.feature_extractor.num_features(), actions)
        self.current_cumulative_reward = 0.0

    def act(self):
        """Execute one action on the world, possibly terminating the episode.

        """
        state = self.domain.get_current_state()
        action = self.choose_action(state)

        self.world.apply_action(action)

        # For the first time step, we won't have received a reward yet.
        # We're just notifying the learner of our starting state and action.
        if self.previousstate is None and self.previousaction is None:
            ()
        else:
            self.value_old = self.update(self.previousstate, self.previousaction, state, action, self.value_old)

        self.previousaction = action
        self.previousstate = state

        state_prime = self.world.get_current_state()
        if self.task.stateisfinal(state_prime):
            self.update(state, action, state_prime, None, self.value_old, terminal=True)

    def update(self, state: State, action: Action, state_prime: State, action_prime: Action, value_old: float,
               terminal=False):
        reward = self.task.reward(state, action, state_prime)

        phi = self.feature_extractor.extract(state, action)

        value = self.value_function.value(phi)

        self._update_traces(phi)
        # Terminal states are defined to have value 0
        if terminal:
            phi_prime = np.zeros(len(phi))
        else:
            phi_prime = self.feature_extractor.extract(state_prime, action_prime)
        value_prime = self.value_function.value(phi_prime)

        delta = reward + self.gamma * value_prime - value
        first = self.alpha * (delta + value - value_old) * self.eligibility
        second = self.alpha * (value - value_old) * phi
        self.value_function.weights += first - second

        value_old = value_prime

        self.current_cumulative_reward += reward

        return value_old

    def _clear_weights(self, weights):
        for i in range(0, len(weights)):
            if weights[i] < 0.000001:
                weights[i] = 0.0
        return weights

    def _update_traces(self, phi):
        e_dot_phi = np.dot(self.eligibility, phi)
        self.eligibility *= self.gamma * self.lamb
        eligibility_target = (self.alpha * self.gamma * self.lamb * e_dot_phi) * phi
        self.eligibility += phi - eligibility_target

    def _eligibility_clear(self):
        for i in range(0, len(self.eligibility)):
            if self.eligibility[i] < 0.00001:
                self.eligibility[i] = 0.0

    def choose_action(self, state) -> Action:
        """Given a state, pick an action according to an epsilon-greedy policy.

        :param state: The state from which to act.
        :return:
        """
        if random.random() < self.epsilon:
            actions = self.domain.get_actions()
            return random.sample(actions, 1)[0]
        else:
            best_actions = self.value_function.bestactions(state, self.feature_extractor)
            return random.sample(best_actions, 1)[0]


    def get_cumulative_reward(self):
        return self.current_cumulative_reward

    def episode_ended(self):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.value_old = 0.0
        self.epsilon *= 0.99999
        self.eligibility = np.zeros(self.feature_extractor.num_features())

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))
