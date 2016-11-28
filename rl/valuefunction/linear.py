from typing import List, Set

import numpy as np

from rl.action import Action
from rl.state import State
from rl.valuefunction import FeatureExtractor


class LinearVFA:
    def __init__(self, num_features, actions: List[Action], per_action_vfa=True, initial_value=0.0):
        self.num_features = num_features
        self.per_action_vfa = per_action_vfa
        self.actions = actions
        self.weights_per_action = dict()
        self.weights = None
        self.reset(initial_value)

    def reset(self, value=0.0):
        if self.per_action_vfa:
            for action in self.actions:
                self.weights_per_action[action] = np.zeros(self.num_features)
        else:
            self.weights = np.zeros(self.num_features)

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
        if self.per_action_vfa:
            weights = self.weights_per_action[action]
        else:
            weights = self.weights
        return weights

    def updateweightsfor(self, weights: np.ndarray, action: Action):
        if self.per_action_vfa:
            self.weights_per_action[action] = weights
        else:
            assert len(weights) == len(self.weights)
            self.weights = weights
