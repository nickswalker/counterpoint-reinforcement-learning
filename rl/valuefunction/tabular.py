from typing import Set

from rl.action import Action
from rl.state import State


class StateActionValueTable:
    def __init__(self, actions: Set[Action]):
        self.table = dict()
        self.actions = actions

    def reset(self):
        self.table = dict()

    def actionvalue(self, state: State, action: Action) -> float:
        # What if its not in the table?
        node = self.table.get(state)
        if node is None:
            entry = dict()
            for a in self.actions:
                entry[a] = 0.0
            self.table[state] = entry

        return self.table[state][action]

    def setactionvalue(self, state: State, action: Action, value: float):
        self.table[state][action] = value

    def bestactions(self, state: State) -> Set[Action]:
        entry = self.table.get(state)
        if entry is None:
            entry = dict()
            for action in self.actions:
                entry[action] = 0.0
        best_actions = []
        best_value = float("-inf")
        for (action, value) in entry.items():
            if value > best_value:
                best_value = value
                best_actions = {action}
            elif value == best_value or abs(value - best_value) < 0.000001:
                best_actions.add(action)

        return best_actions
