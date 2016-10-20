import random

import numpy as np

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task
from rl.valuefunction import FeatureExtractor
from rl.valuefunction.linear import LinearVFA


class TrueOnlineSarsaLambda(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.6,
                 gamma=0.95, lamb=0.95, expected=False):
        """
        :param domain: The world the agent is placed in.
        :param task: The task in the world, which defines the reward function.
        """
        super().__init__(domain, task)
        self.world = domain
        self.task = task
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.expected = expected
        self.previousaction = None
        self.previousstate = None
        self.value_old = 0.0

        example_state = domain.get_current_state()
        actions = domain.get_actions(example_state)
        self.feature_extractor = feature_extractor
        self.eligibility = np.zeros(self.feature_extractor.number_of_features)
        self.value_function = LinearVFA(self.feature_extractor.number_of_features, actions)
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

        state_weights = np.array(self.value_function.weightsfor(action))
        state_features = np.array(self.feature_extractor.extract(state, action))

        value_prime = np.dot(state_weights, state_features)
        value = np.dot(state_weights, state_features)

        self._update_traces(state_features)
        # Terminal states are defined to have value 0
        if terminal:
            value_prime = 0
        else:
            if self.expected:
                value_prime = self.expected_value(state_prime)

            else:
                state_prime_features = np.array(self.feature_extractor.extract(state_prime, action_prime))
                value_prime = np.dot(state_weights, state_prime_features)

        delta = reward + self.gamma * value_prime - value
        weights_target = delta * self.eligibility + self.alpha * (
            value - np.dot(state_weights, state_features)) * state_features
        updated_weights = state_weights + weights_target

        # updated_weights = self._clear_weights(updated_weights)
        self.value_function.updateweightsfor(updated_weights, action)
        value_old = self.value_function.actionvalue(state_features, action)

        self.current_cumulative_reward += reward

        self._eligibility_clear()
        return value_old

    def _clear_weights(self, weights):
        for i in range(0, len(weights)):
            if weights[i] < 0.000001:
                weights[i] = 0.0
        return weights

    def _update_traces(self, state_features):
        discounted_eligibility = self.gamma * self.lamb * self.eligibility
        eligibility_target = self.alpha * (1 - self.gamma * self.lamb * np.dot(self.eligibility,
                                                                               state_features)) * state_features
        self.eligibility = discounted_eligibility + eligibility_target

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
            actions = self.domain.get_actions(state)
            return random.sample(actions, 1)[0]
        else:
            best_actions = self.value_function.bestactions(state, self.feature_extractor)
            return random.sample(best_actions, 1)[0]

    def expected_value(self, state):
        actions = self.domain.get_actions(state)
        expectation = 0.0
        best_actions = self.value_function.bestactions(state)
        num_best_actions = len(best_actions)
        nonoptimal_mass = self.epsilon

        if num_best_actions > 0:
            a_best_action = random.sample(best_actions, 1)[0]
            greedy_mass = (1.0 - self.epsilon)
            expectation += greedy_mass * self.value_function.actionvalue(state, a_best_action)
        else:
            nonoptimal_mass = 1.0

        if nonoptimal_mass > 0.0:
            # No best action, equiprobable random policy
            total_value = 0.0
            for action in actions:
                total_value += self.value_function.actionvalue(state, action)
            expectation += nonoptimal_mass * total_value / len(actions)

        return expectation

    def get_cumulative_reward(self):
        return self.current_cumulative_reward

    def episode_ended(self):
        # Observe final transition if needed
        self.current_cumulative_reward = 0.0
        self.previousaction = None
        self.previousstate = None
        self.value_old = 0.0

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))
