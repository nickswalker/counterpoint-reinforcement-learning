from typing import Tuple

from abjad.tools.durationtools import Duration

from counterpoint.composition_parameters import CompositionParameters
from counterpoint.hashable_note import HashableNote
from rl.action import Action
from rl.state import State


class CompositionState(State):
    def __init__(self, preceding_duration: Duration, voices: Tuple[Tuple[HashableNote]],
                 composition_parameters: CompositionParameters):
        self.composition_parameters = composition_parameters
        self.preceding_duration = preceding_duration
        self.voices = voices

    def __eq__(self, other):
        if isinstance(other, CompositionState):
            return self.preceding_duration == other.preceding_duration and self.voices == other.voices
        return False

    def __hash__(self):
        return hash((self.preceding_duration, self.voices))

    def __str__(self):
        return str(self.voices) + " " + str(self.preceding_duration)


class PositionIndependentCompositionState(State):
    def __eq__(self, other):
        if isinstance(other, PositionIndependentCompositionState):
            return self.voices == other.voices
        return False

    def __hash__(self):
        return hash(self.voices)

    def __str__(self):
        return str(self.voices) + " " + str(self.preceding_duration)


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

    def __str__(self):
        return str(self.notes_per_voice)
