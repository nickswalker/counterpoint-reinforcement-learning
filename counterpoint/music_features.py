from enum import Enum
from typing import List

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.scoretools import Note
from abjad.tools.scoretools import Voice

from counterpoint.composition_environment import CompositionState
from rl.valuefunction.FeatureExtractor import StateFeatureExtractor


class Motion(Enum):
    similar = 0
    parallel = 1
    contrary = 2
    oblique = 3


class MusicFeatureExtractor(StateFeatureExtractor):
    def __init__(self, num_pitches_per_voice: List[int], history_length: int):
        self.num_voices = len(num_pitches_per_voice)
        self.pitches_per_voice = num_pitches_per_voice
        self.history_length = history_length

    def extract(self, state: CompositionState) -> List[float]:
        features = []
        for i in range(0, self.num_voices):
            for j in range(0, self.history_length):
                item = state.voices[i][j]
                if isinstance(item, Note):
                    # One hot last pitch per voice
                    pitch = item.written_pitch
                    index = state.composition_parameters.pitch_indices[i][pitch]
                    section = [0] * self.pitches_per_voice[i]
                    section[index] = 1
                    features += section
                else:
                    features += [0] * self.pitches_per_voice[i]

        return features

    def num_features(self) -> int:
        return sum(self.pitches_per_voice) * self.history_length

    @staticmethod
    def last_n_intervals(n: int, current_duration: Duration, uppervoice: Voice, lowervoice: Voice) -> \
            List[
                NamedInterval]:
        all_harmonic = []
        all_melodic = [[], []]

        u = l = 0
        upper_duration = lower_duration = Duration(0)
        previous_lower = previous_upper = None
        while upper_duration < current_duration and lower_duration < current_duration:
            upper = uppervoice[u]
            lower = lowervoice[l]
            upper_duration += upper.written_duration
            lower_duration += lower.written_duration

            interval = NamedInterval.from_pitch_carriers(lower, upper)
            all_harmonic.append(interval)
            if upper_duration == lower_duration:
                u += 1
                l += 1

                low_interval = interval_or_none(previous_lower, lower)
                if low_interval is not None:
                    all_melodic[1].append(low_interval)
                upper_interval = interval_or_none(previous_upper, upper)
                if upper_interval is not None:
                    all_melodic[0].append(upper_interval)
                previous_lower = lower
                previous_upper = upper
            elif upper_duration < lower_duration:
                u += 1
                previous_upper = upper

                upper_interval = interval_or_none(previous_upper, upper)
                if upper_interval is not None:
                    all_melodic[0].append(upper_interval)
            elif lower_duration < upper_duration:
                l += 1
                previous_lower = lower
                low_interval = interval_or_none(previous_lower, lower)
                if low_interval is not None:
                    all_melodic[1].append(low_interval)

        num_to_return = min(n, len(all_harmonic))
        melodic = [voice[-num_to_return:] for voice in all_melodic]
        return all_harmonic[-num_to_return:], melodic

    @staticmethod
    def characterize_relative_motion(upper_motion: NamedInterval, lower_motion: NamedInterval) -> Motion:
        # NOTE: handle double unison
        if upper_motion.direction_string == lower_motion.direction_string:
            if upper_motion.interval_string == lower_motion.interval_string:
                return Motion.parallel
            else:
                return Motion.similar
        else:
            return Motion.contrary


def interval_or_none(first, second):
    if first is None or second is None:
        return None
    else:
        return NamedInterval.from_pitch_carriers(first, second)
