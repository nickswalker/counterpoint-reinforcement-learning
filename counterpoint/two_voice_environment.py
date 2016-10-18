from typing import Set, List

from abjad import *
from abjad.tools.metertools import Meter

from rl.action import Action
from rl.domain import Domain
from rl.state import State


class CompositionState(State):
    def __init__(self, beat: int, voices: List[Note]):
        self.beat = beat
        self.voices = voices


class CompositionAction(Action):
    def __init__(self, notes_per_voice: List[Note]):
        self.notes_per_voice = notes_per_voice


class CompositionEnvironment(Domain):
    def __init__(self, number_of_voices_to_generate: int, given_voices: List[Voice], meter: Meter, key: KeySignature):
        assert meter.is_simple()
        self.key = key
        self.meter = meter
        self.given_voices = given_voices
        self.number_of_voices_to_generate = number_of_voices_to_generate

        self.voices = [Voice("") * number_of_voices_to_generate]
        self.current_beat = 0
        self.total_beats = meter.denominator

    def get_actions(self, state: State) -> Set[Action]:
        pass

    def get_current_state(self) -> State:
        return CompositionState(self.current_beat, )

    def apply_action(self, action: CompositionAction):
        pass

    def reset(self):
        self.lines =
        pass
