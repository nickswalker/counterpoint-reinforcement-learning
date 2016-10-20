import itertools
from typing import Set, List, Tuple

from abjad import *
from abjad.tools.metertools import Meter

from counterpoint.HashableNote import HashableNote
from rl.action import Action
from rl.domain import Domain
from rl.state import State


class CompositionState(State):
    def __init__(self, beat: int, voices: List[Note]):
        self.beat = beat
        self.voices = voices


class CompositionAction(Action):
    def __init__(self, notes_per_voice: Tuple[Note]):
        self.notes_per_voice = notes_per_voice

    def __eq__(self, other):
        if isinstance(other, CompositionAction):
            match = True
            for note, other_note in zip(self.notes_per_voice, other.notes_per_voice):
                match = match and note == other_note
            return match
        return False

    def __hash__(self):
        return hash(self.notes_per_voice)



class CompositionEnvironment(Domain):
    def __init__(self, number_of_voices_to_generate: int, given_voices: List[Voice], meter: Meter, key: KeySignature):
        assert meter.is_simple
        self.key = key
        self.meter = meter
        self.given_voices = given_voices
        self.number_of_voices_to_generate = number_of_voices_to_generate

        self.voices = [Staff("") for i in range(number_of_voices_to_generate)]
        self.current_beat = 0

    def get_actions(self, state: State) -> Set[Action]:
        possible_pitches_per_voice = []
        for i in range(self.number_of_voices_to_generate):
            possible_pitches_per_voice.append([NamedPitch("C4")])
        possible_durations_per_voice = [[1] for _ in range(self.number_of_voices_to_generate)]
        actions = []

        notes_per_voice = []
        for i in range(self.number_of_voices_to_generate):
            notes_per_voice.append([])
            for pitch, duration in itertools.product(possible_pitches_per_voice[i], possible_durations_per_voice[i]):
                notes_per_voice[i].append(HashableNote(pitch, duration))

        for notes in itertools.product(*notes_per_voice):
            actions.append(CompositionAction(notes))
        return actions

    def get_current_state(self) -> State:
        notes_in_voices = []
        for voice in self.voices:
            if len(voice) > 0:
                notes_in_voices.append(voice[len(voice) - 1])
            else:
                notes_in_voices.append(None)
        for voice in self.given_voices:
            notes_in_voices.append(voice[self.current_beat])
        return CompositionState(self.current_beat, tuple(notes_in_voices))

    def apply_action(self, action: CompositionAction):
        self.current_beat += 1
        for voice, note in zip(self.voices, action.notes_per_voice):
            voice.append(note)

    def reset(self):
        self.voices = [Staff("") for i in range(self.number_of_voices_to_generate)]
        self.current_beat = 0
