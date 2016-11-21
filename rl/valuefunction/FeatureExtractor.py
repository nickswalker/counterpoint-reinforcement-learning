from abc import abstractmethod
from typing import List

from rl.action import Action
from rl.state import State


class StateActionFeatureExtractor:
    @abstractmethod
    def num_features(self) -> int:
        pass

    @abstractmethod
    def extract(self, state: State, action: Action) -> List[float]:
        pass


class StateFeatureExtractor:
    @abstractmethod
    def num_features(self) -> int:
        pass

    @abstractmethod
    def extract(self, state: State) -> List[float]:
        pass
