import random

import numpy as np

from rl.action import Action
from rl.agent.agent import Agent
from rl.domain import Domain
from rl.state import State
from rl.task import Task
from rl.valuefunction import FeatureExtractor
from rl.valuefunction.CMACValueFunction import CMACValueFunction


class SarsaCMAC(Agent):
    def __init__(self, domain: Domain, task: Task, feature_extractor: FeatureExtractor, epsilon=0.1, alpha=0.6,
                 gamma=0.95, name="Sarsa"):
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
        self.previousaction = None
        self.previousstate = None

        actions = domain.get_actions()
        self.feature_extractor = feature_extractor
        self.value_function = CMACValueFunction(self.feature_extractor.num_features(), actions, alpha)

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
            self.update(self.previousstate, self.previousaction, state, action)

        self.previousaction = action
        self.previousstate = state

        state_prime = self.world.get_current_state()
        if self.task.stateisfinal(state_prime):
            self.update(state, action, state_prime, None, terminal=True)

    def update(self, state: State, action: Action, state_prime: State, action_prime: Action,
               terminal=False):
        reward = self.task.reward(state, action, state_prime)

        phi = self.feature_extractor.extract(state, action)

        value = self.value_function.value(phi)

        # Terminal states are defined to have value 0
        if terminal:
            phi_prime = np.zeros(len(phi))
        else:
            phi_prime = self.feature_extractor.extract(state_prime, action_prime)
        value_prime = self.value_function.value(phi_prime)

        self.value_function.train(phi, reward + self.gamma * value_prime)
        self.current_cumulative_reward += reward

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
        self.epsilon *= 0.99999

    def logepisode(self):
        print("Episode reward: " + str(self.current_cumulative_reward))
