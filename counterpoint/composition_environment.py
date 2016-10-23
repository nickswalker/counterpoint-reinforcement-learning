import itertools
from typing import Set, List, Tuple

from abjad import *
from abjad.tools.metertools import Meter

from counterpoint.HashableNote import HashableNote
from rl.action import Action
from rl.domain import Domain
from rl.state import State


class CompositionState(State):
    def __init__(self, preceding_duration: Duration, voices: Tuple[Note]):
        self.preceding_duration = preceding_duration
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
    def __init__(self, given_voices: List[Voice],
                 desired_voices: List[Tuple[str, Tuple[NamedPitch, NamedPitch]]], meter: Meter, key: KeySignature):
        assert meter.is_simple
        self.key = key
        self.meter = meter
        self.given_voices = given_voices
        self.number_of_voices_to_generate = len(desired_voices)
        self.desired_voices = desired_voices

        self.voices = [Voice("", name=name) for name, range in desired_voices]
        self.current_duration = Duration(0)

        just_ranges = [range for name, range in desired_voices]
        self.actions = self.generate_actions(just_ranges)

    def get_actions(self, state: State) -> Set[Action]:
        return self.actions

    def get_current_state(self) -> State:
        notes_in_voices = []
        for voice in self.voices:
            if len(voice) > 0:
                notes_in_voices.append(voice[len(voice) - 1])
            else:
                notes_in_voices.append(None)
        for voice in self.given_voices:
            notes_in_voices.append(voice[len(voice) - 1])
        return CompositionState(self.current_duration, tuple(notes_in_voices))

    def apply_action(self, action: CompositionAction):
        for voice, note in zip(self.voices, action.notes_per_voice):
            voice.append(note)
        self.current_duration = inspect_(voice).get_duration()

    def reset(self):
        self.voices = [Staff("") for i in range(self.number_of_voices_to_generate)]
        self.current_duration = Duration(0)

    def generate_actions(self, voice_ranges):
        possible_pitches_per_voice = []
        for i in range(self.number_of_voices_to_generate):
            possible_pitches_per_voice.append([])
            current_pitch = voice_ranges[i][0].numbered_pitch
            while current_pitch <= voice_ranges[i][1]:
                possible_pitches_per_voice[i].append(current_pitch)
                current_pitch = current_pitch + 1
        possible_durations_per_voice = [[Duration(1, 4)] for _ in range(self.number_of_voices_to_generate)]
        actions = []

        notes_per_voice = []
        for i in range(self.number_of_voices_to_generate):
            notes_per_voice.append([])
            for pitch, duration in itertools.product(possible_pitches_per_voice[i], possible_durations_per_voice[i]):
                notes_per_voice[i].append(HashableNote(pitch, duration))

        for notes in itertools.product(*notes_per_voice):
            actions.append(CompositionAction(notes))
        return actions
