from abjad.tools.durationtools.Duration import Duration

from counterpoint.composition_environment import CompositionEnvironment, CompositionState
from rl.task import Task


class SpeciesOneCounterpoint(Task):
    def __init__(self, domain: CompositionEnvironment, desired_duration=Duration(3)):
        super().__init__(domain)
        self.domain = domain
        self.desired_duration = desired_duration

    def stateisfinal(self, state: CompositionState):
        if self.desired_duration == state.preceding_duration:
            return True
        return False

    def reward(self, state, action, state_prime):
        return -1
