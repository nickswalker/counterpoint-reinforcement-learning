from enum import Enum
from typing import List

import numpy as np
from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.scoretools import Note

from counterpoint.composition_environment import CompositionState
from rl.valuefunction.FeatureExtractor import StateFeatureExtractor


class RelativeMotion(Enum):
    similar = 0
    parallel = 1
    contrary = 2
    oblique = 3
    none = 4


class Motion(Enum):
    step_up = 0
    step_down = 1
    leap_up = 2
    leap_down = 3
    none = 4


class MusicFeatureExtractor(StateFeatureExtractor):
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


def characterize_relative_motion(upper_motion: NamedInterval, lower_motion: NamedInterval) -> RelativeMotion:
    if upper_motion.direction_string == lower_motion.direction_string:
        if upper_motion.semitones == 0 and lower_motion.semitones == 0:
            return RelativeMotion.none
        elif upper_motion.interval_string == lower_motion.interval_string:
            return RelativeMotion.parallel
        else:
            return RelativeMotion.similar
    else:
        return RelativeMotion.contrary


def interval_or_none(first, second):
    if first is None or second is None:
        return None
    else:
        return NamedInterval.from_pitch_carriers(first, second)
