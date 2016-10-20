from abc import abstractmethod
from typing import List

from rl.action import Action
from rl.state import State


class FeatureExtractor:
    @abstractmethod
    def num_features(self) -> int:
        pass

    @abstractmethod
    def extract(self, state: State, action: Action) -> List[float]:
        pass
