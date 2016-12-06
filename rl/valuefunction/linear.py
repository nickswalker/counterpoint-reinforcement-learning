from typing import List, Set

import numpy as np

from rl.action import Action
from rl.state import State
from rl.valuefunction import FeatureExtractor


class PerActionLinearVFA:
    def __init__(self, num_features, actions: List[Action], initial_value=0.0):
        self.num_features = num_features
        self.actions = actions
        self.weights_per_action = dict()
        self.weights = None
        self.reset(initial_value)

    def reset(self, value=0.0):
        for action in self.actions:
            self.weights_per_action[action] = np.zeros(self.num_features)


    def actionvalue(self, features: np.ndarray, action: Action) -> float:
        return np.dot(self.weightsfor(action), features)

    def statevalue(self, features: List[float]):
        raise Exception()

    def bestactions(self, state: State, extractor: FeatureExtractor) -> Set[Action]:
        best_actions = []
        best_value = float("-inf")
        for action in self.actions:
            state_features = extractor.extract(state)
            value = self.actionvalue(state_features, action)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return best_actions

    def weightsfor(self, action: Action) -> np.ndarray:
        return self.weights_per_action[action]

    def updateweightsfor(self, weights: np.ndarray, action: Action):
        self.weights_per_action[action] = weights


class LinearVFA:
    def __init__(self, num_features, actions: List[Action], initial_value=0.0):
        self.num_features = num_features
        self.actions = actions
        self.weights = np.zeros(num_features)
        self.reset(initial_value)

    def reset(self, value=0.0):
        self.weights = np.zeros(self.num_features)

    def value(self, features: np.ndarray) -> float:
        return np.dot(self.weights, features)

    def bestactions(self, state: State, extractor: FeatureExtractor) -> Set[Action]:
        best_actions = []
        best_value = float("-inf")
        for action in self.actions:
            phi = extractor.extract(state, action)
            value = self.value(phi)
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        return best_actions

    def updateweights(self, weights: np.ndarray):
        assert len(weights) == len(self.weights)
        self.weights = weights
