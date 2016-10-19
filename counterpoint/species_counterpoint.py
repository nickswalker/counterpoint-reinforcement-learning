from typing import List
from typing import Tuple

from abjad.tools.pitchtools import NamedPitch

from counterpoint.composition_environment import CompositionEnvironment, CompositionState
from rl.task import Task


class SpeciesOneCounterpoint(Task):
    def __init__(self, domain: CompositionEnvironment, voice_ranges: List[Tuple[NamedPitch, NamedPitch]], measures=5):
        self.domain = domain
        self.voice_ranges = voice_ranges
        self.measures = measures

    def stateisfinal(self, state: CompositionState):
        if self.domain.total_beats == state
            pass

    def reward(self, state, action, state_prime):
        pass
