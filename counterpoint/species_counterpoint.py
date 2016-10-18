from typing import List
from typing import Tuple

from abjad.tools.pitchtools import NamedPitch

from rl.task import Task


class SpeciesOneCounterpoint(Task):
    def __init__(self, measures=5, voice_ranges=List[Tuple[NamedPitch, NamedPitch]]):
        self.voice_ranges = voice_ranges
        self.measures = measures

    def stateisfinal(self, state):
        pass

    def reward(self, state, action, state_prime):
        pass
