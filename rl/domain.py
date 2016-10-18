from typing import Set

from rl.action import Action
from rl.state import State


class Domain():
    def get_actions(self, state: State) -> Set[Action]:
        raise NotImplementedError("Should have implemented this")

    def apply_action(self, action: Action):
        raise NotImplementedError("Should have implemented this")

    def reset(self):
        raise NotImplementedError("Should have implemented this")

    def get_current_state(self) -> State:
        raise NotImplementedError("Should have implemented this")
