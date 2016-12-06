from typing import List, Set

import numpy as np

from rl.action import Action
from rl.state import State
from rl.valuefunction import FeatureExtractor
from rl.valuefunction.CMAC import CMAC


class CMACValueFunction:
    def __init__(self, num_features, actions: List[Action], initial_value=0.0, beta=0.01):
        self.num_features = num_features
        self.actions = actions
        self.beta = beta
        self.cmac = CMAC(32, 1, beta)
        self.reset()

    def reset(self):
        self.cmac = CMAC(32, 1, self.beta)

    def value(self, features: np.ndarray) -> float:
        return self.cmac.eval(features)

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

    def train(self, features: np.ndarray, response):
        self.cmac.train(features, response)
