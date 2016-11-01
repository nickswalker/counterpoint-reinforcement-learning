import itertools
from typing import Set, List, Tuple

from abjad import *
from abjad.tools.metertools import Meter
from abjad.tools.pitchtools.PitchRange import PitchRange
from abjad.tools.tonalanalysistools.Scale import Scale

from counterpoint.HashableNote import HashableNote
from rl.action import Action
from rl.domain import Domain
from rl.state import State


class CompositionState(State):
    def __init__(self, preceding_duration: Duration, voices: Tuple[HashableNote]):
        self.preceding_duration = preceding_duration
        self.voices = voices


class CompositionAction(Action):
    def __init__(self, notes_per_voice: Tuple[HashableNote]):
        self.notes_per_voice = notes_per_voice
        for note in notes_per_voice:
            assert note is not None

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
                 desired_voices: List[Tuple[str, PitchRange]], meter: Meter, scale: Scale):
        assert meter.is_simple
        self.scale = scale
        self.meter = meter
        self.given_voices = given_voices
        self.number_of_voices_to_generate = len(desired_voices)
        self.desired_voices = desired_voices

        self.voices = [Voice("", name=name) for name, pitch_range in desired_voices]
        self.current_duration = Duration(0)

        self.actions = self.generate_actions()

    def get_actions(self, state: State) -> Set[Action]:
        return self.actions

    def get_current_state(self) -> State:
        notes_in_voices = []
        for voice in self.voices:
            previous_note = None
            if len(voice) > 0:
                previous_note = voice[len(voice) - 1]
            previous_note = HashableNote(str(previous_note)) if previous_note is not None else None
            notes_in_voices.append(tuple([previous_note]))

        for given_voice in self.given_voices:
            total_duration = Duration(0)
            previous_note = None
            next_note = None

            if self.current_duration == Duration(0):
                previous_note = None
                next_note = given_voice[0]
            else:
                previous_note = given_voice[0]
                for next_note in given_voice[1:]:
                    total_duration += next_note.written_duration
                    if total_duration >= self.current_duration:
                        break
                    previous_note = next_note

            previous_note = HashableNote(str(previous_note)) if previous_note is not None else None
            next_note = HashableNote(str(next_note)) if next_note is not None else None
            notes_in_voices.append((previous_note, next_note))
        return CompositionState(self.current_duration, tuple(notes_in_voices))

    def apply_action(self, action: CompositionAction):
        for voice, note in zip(self.voices, action.notes_per_voice):
            # Hashable notes are only for state representation
            voice.append(Note(str(note)))
        self.current_duration = inspect_(self.voices[0]).get_duration()

    def reset(self):
        self.voices = [Voice("") for i in range(self.number_of_voices_to_generate)]
        self.current_duration = Duration(0)

    def generate_actions(self):
        possible_pitches_per_voice = []
        for name, voice_range in self.desired_voices:
            pitch_set = self.scale.create_named_pitch_set_in_pitch_range(voice_range)
            possible_pitches_per_voice.append(pitch_set)

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
