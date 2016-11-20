from enum import Enum
from typing import List

from abjad.tools.durationtools import Duration
from abjad.tools.pitchtools import NamedInterval
from abjad.tools.scoretools import Voice

from counterpoint.composition_environment import CompositionState
from rl.valuefunction.FeatureExtractor import FeatureExtractor


class Motion(Enum):
    similar = 0
    parallel = 1
    contrary = 2
    oblique = 3


class MusicFeatureExtractor(FeatureExtractor):
    def extract(self, state: CompositionState) -> List[float]:
        current_upper_pitch = state.voices[0][0].written_pitch
        current_lower_pitch = state.voices[1][0].written_pitch

        harmonic_intervals, melodic_intervals = MusicFeatureExtractor.last_n_intervals(3, self.domain.current_duration,
                                                                                       self.domain.voices[0],
                                                                                       self.domain.given_voices[0])
        features = []
        pass

    def num_features(self) -> int:
        return 1

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
