import itertools
from typing import Set, List

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.scoretools.Note import Note
from abjad.tools.scoretools.Rest import Rest
from abjad.tools.scoretools.Voice import Voice
from abjad.tools.topleveltools.inspect_ import inspect_

from counterpoint.composition_parameters import CompositionParameters
from counterpoint.composition_state_action import CompositionAction, CompositionState
from counterpoint.hashable_note import HashableNote
from rl.action import Action
from rl.domain import Domain
from rl.state import State


class CompositionEnvironment(Domain):
    def __init__(self, composition_parameters: CompositionParameters, given_voices: List[Voice] = list(),
                 history_length=2):
        self.given_voices = given_voices
        self.composition_parameters = composition_parameters
        self.history_length = history_length

        self.voices = [Voice("", name=name) for name, pitch_range in self.composition_parameters.desired_voices]
        self.current_duration = Duration(0)

        actionslist, index_to_action, action_to_index = self._generate_actions()
        self.actions = actionslist
        self.index_to_action = index_to_action
        self.action_to_index = action_to_index

    def get_actions(self) -> Set[Action]:
        return self.actions

    def get_current_state(self) -> State:
        k = self.history_length
        notes_in_voices = []

        for voice in self.voices:
            lookback = min(len(voice), k)
            history = list(voice[-lookback:])
            if len(history) < k:
                shortfall = k - len(history)
                fill = [Rest(Duration(1, 4))] * shortfall
                history = fill + history

            notes_in_voices.append(tuple(history))

        # Find the closest index
        for given_voice in self.given_voices:
            total_duration = Duration(0)
            i = 0
            for i in range(0, len(given_voice)):
                next_note = given_voice[i]
                if total_duration + next_note.written_duration >= self.current_duration:
                    break
                total_duration += next_note.written_duration

            history = list(given_voice[i - k:i])
            if len(history) < k:
                shortfall = k - len(history)
                fill = [Rest(Duration(1, 4))] * shortfall
                history = fill + history

            notes_in_voices.append(tuple(history))

        return CompositionState(self.current_duration, tuple(notes_in_voices), self.composition_parameters)

    def apply_action(self, action: CompositionAction):
        for voice, note in zip(self.voices, action.notes_per_voice):
            # Hashable notes are only for state representation
            voice.append(Note(str(note)))
        self.current_duration = inspect_(self.voices[0]).get_duration()

    def reset(self):
        self.voices = [Voice("") for _ in range(self.composition_parameters.number_of_voices_to_generate)]
        self.current_duration = Duration(0)

    def _generate_actions(self):
        possible_pitches_per_voice = []
        for name, voice_range in self.composition_parameters.desired_voices:
            pitch_set = self.composition_parameters.scale.create_named_pitch_set_in_pitch_range(voice_range)
            possible_pitches_per_voice.append(pitch_set)

        possible_durations_per_voice = [[Duration(1, 4)] for _ in
                                        range(self.composition_parameters.number_of_voices_to_generate)]
        actions = []

        notes_per_voice = []
        for i in range(self.composition_parameters.number_of_voices_to_generate):
            notes_per_voice.append([])
            for pitch, duration in itertools.product(possible_pitches_per_voice[i], possible_durations_per_voice[i]):
                notes_per_voice[i].append(HashableNote(pitch, duration))

        index_to_action = dict()
        action_to_index = dict()
        i = 0
        for notes in itertools.product(*notes_per_voice):
            action = CompositionAction(notes)
            actions.append(action)
            index_to_action[i] = action
            action_to_index[action] = i
            i += 1
        return actions, index_to_action, action_to_index
