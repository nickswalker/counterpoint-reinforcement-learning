import itertools
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
        possible_pitches = [NamedPitch("C4")]
        possible_durations = [1]
        actions = []
        for pitch, duration in itertools.product(possible_pitches, possible_durations):
            actions.append(CompositionAction(Note(pitch, duration)))
        return actions

    def get_current_state(self) -> State:
        notes_in_voices = []
        for voice in self.voices:
            notes_in_voices.append(voice.last())
        for voice in self.given_voices:
            notes_in_voices.append(voice.get(self.current_beat))
        return CompositionState(self.current_beat, notes_in_voices)

    def apply_action(self, action: CompositionAction):
        for voice, note in zip(self.voices, action.notes_per_voice):
            voice.append(note)

    def reset(self):
        for voice in self.voices:
            voice.clear()
        pass
