from typing import List

import numpy as np
from abjad.tools.scoretools import Note

from counterpoint.composition_environment import CompositionState
from counterpoint.composition_state_action import CompositionAction
from rl.valuefunction.FeatureExtractor import StateFeatureExtractor, StateActionFeatureExtractor


class MusicStateFeatureExtractor(StateFeatureExtractor):
    def __init__(self, num_pitches_per_voice: List[int], history_length: int, position_invariant: bool = False):
        self.num_voices = len(num_pitches_per_voice)
        self.position_invariant = position_invariant
        # Include a null class for unpopulated features
        self.max_beats = 12
        self.options_per_voice = [num_pitches + 1 for num_pitches in num_pitches_per_voice]
        self.history_length = history_length

    def extract(self, state: CompositionState) -> np.array:
        features = np.zeros(self.num_features())
        if len(features) == 1:
            features[0] = 1
            return features
        position = 0
        for i in range(0, self.num_voices):
            for j in range(0, self.history_length):
                item = state.voices[i][j]
                if isinstance(item, Note):
                    # One pitch per voice
                    pitch = item.written_pitch
                    index = state.composition_parameters.pitch_indices[i][pitch]
                    features[position + index] = 1
                else:
                    index = self.options_per_voice[i] - 1
                    features[position + index] = 1
                position += self.options_per_voice[i]

        if not self.position_invariant:
            beat = int(float(state.preceding_duration) / 1.00)
            features[position + beat] = 1
        return features

    def num_features(self) -> int:
        if self.position_invariant:
            num = sum(self.options_per_voice) * self.history_length
        else:
            num = sum(self.options_per_voice) * self.history_length + self.max_beats
        #
        if num == 0:
            return 1
        return num


class MusicStateActionFeatureExtractor(StateActionFeatureExtractor):
    def __init__(self, num_pitches_per_voice: List[int], history_length: int, position_invariant: bool = False):
        self.num_voices = len(num_pitches_per_voice)
        self.position_invariant = position_invariant
        # Include a null class for unpopulated features
        self.max_beats = 12
        self.options_per_voice = [num_pitches + 1 for num_pitches in num_pitches_per_voice]
        self.history_length = history_length

    def extract(self, state: CompositionState, action: CompositionAction) -> np.array:
        features = np.zeros(self.num_features())
        if len(features) == 1:
            features[0] = 1
            return features
        position = 0
        for i in range(0, self.num_voices):
            for j in range(0, self.history_length):
                item = state.voices[i][j]
                if isinstance(item, Note):
                    # One pitch per voice
                    pitch = item.written_pitch
                    index = state.composition_parameters.pitch_indices[i][pitch]
                    features[position + index] = 1
                else:
                    index = self.options_per_voice[i] - 1
                    features[position + index] = 1
                position += self.options_per_voice[i]

        for i in range(0, self.num_voices):
            item = action.notes_per_voice[i]
            if isinstance(item, Note):
                # One pitch per voice
                pitch = item.written_pitch
                index = state.composition_parameters.pitch_indices[i][pitch]
                features[position + index] = 1
            position += self.options_per_voice[i]

        if not self.position_invariant:
            beat = int(float(state.preceding_duration) / 1.00)
            features[position + beat] = 1
        return features

    def num_features(self) -> int:
        if self.position_invariant:
            num = sum(self.options_per_voice) * (self.history_length + 1)
        else:
            num = sum(self.options_per_voice) * (self.history_length + 1) + self.max_beats
        #
        if num == 0:
            return 1
        return num
