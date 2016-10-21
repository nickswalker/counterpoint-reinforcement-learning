from counterpoint.composition_environment import CompositionEnvironment, CompositionState
from rl.task import Task


class SpeciesOneCounterpoint(Task):
    def __init__(self, domain: CompositionEnvironment, measures=5):
        self.domain = domain
        self.measures = measures
        self.total_beats = measures * domain.meter.denominator

    def stateisfinal(self, state: CompositionState):
        if self.total_beats == state.beat:
            return True
        return False

    def reward(self, state, action, state_prime):
        return -1
