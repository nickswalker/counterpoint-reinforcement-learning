from abc import abstractmethod
from enum import Enum
from typing import List

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.scoretools import Voice

from counterpoint import constants
from counterpoint.composition_environment import CompositionEnvironment, CompositionState, CompositionAction
from rl.task import Task


class Motion(Enum):
    similar = 0
    parallel = 1
    contrary = 2
    oblique = 3


class CounterpointTask(Task):
    def __init__(self, domain: CompositionEnvironment, desired_duration=Duration(3)):
        super().__init__(domain)
        self.domain = domain
        self.desired_duration = desired_duration

    def stateisfinal(self, state: CompositionState):
        if self.desired_duration <= state.preceding_duration:
            return True
        return False

    @abstractmethod
    def reward(self, state, action, state_prime):
        return 0


class SpeciesOneCounterpoint(CounterpointTask):
    def __init__(self, domain: CompositionEnvironment, desired_duration=Duration(3)):
        super().__init__(domain, desired_duration)

    def reward(self, state: CompositionState, action: CompositionAction, state_prime: CompositionState):
        current_reward = 0

        current_upper_pitch = state_prime.voices[0][0]
        current_lower_pitch = state_prime.voices[1][0]

        harmonic_intervals, melodic_intervals = self.last_n_intervals(3, self.domain.current_duration,
                                                                      self.domain.voices[0],
                                                                      self.domain.given_voices[0])

        # No voice crossing
        if current_lower_pitch > current_upper_pitch or current_upper_pitch < current_lower_pitch:
            current_reward -= 1

        # There is always at least one harmonic interval
        harmonic_interval = harmonic_intervals[-1]
        # No intervals greater than a 12th
        if harmonic_interval > NamedInterval("P12"):
            current_reward -= 2
        # Tenth is pushing it
        elif harmonic_interval > NamedInterval("M10"):
            current_reward -= 1

        # Is this the first beat?
        if state.preceding_duration == Duration(0):
            # First beat should be a perfect interval
            if harmonic_interval not in constants.perfect_intervals:
                current_reward -= 1
            return current_reward

        # No unisons in the middle of the piece
        if harmonic_interval is NamedInterval("P1"):
            current_reward -= 1

        # If there's more than one note then we can look at the melodic intervals
        contrapuntal_melodic_interval = melodic_intervals[0][-1]
        cantus_melodic_interval = melodic_intervals[0][-1]
        if contrapuntal_melodic_interval is NamedInterval("P1"):
            current_reward -= 1
        elif contrapuntal_melodic_interval in constants.dissonant_intervals:
            current_reward -= 1

        parallel_motion = contrapuntal_melodic_interval.direction_number == cantus_melodic_interval.direction_number and contrapuntal_melodic_interval == cantus_melodic_interval

        if harmonic_interval in constants.dissonant_intervals:
            current_reward -= 1
        if parallel_motion:
            current_reward -= 1

        return current_reward

    def is_step(self, interval: NamedInterval):
        pass

    def last_n_intervals(self, n: int, current_duration: Duration, uppervoice: Voice, lowervoice: Voice) -> \
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

                low_interval = self.interval_or_none(previous_lower, lower)
                if low_interval is not None:
                    all_melodic[1].append(low_interval)
                upper_interval = self.interval_or_none(previous_upper, upper)
                if upper_interval is not None:
                    all_melodic[0].append(upper_interval)
                previous_lower = lower
                previous_upper = upper
            elif upper_duration < lower_duration:
                u += 1
                previous_upper = upper

                upper_interval = self.interval_or_none(previous_upper, upper)
                if upper_interval is not None:
                    all_melodic[0].append(upper_interval)
            elif lower_duration < upper_duration:
                l += 1
                previous_lower = lower
                low_interval = self.interval_or_none(previous_lower, lower)
                if low_interval is not None:
                    all_melodic[1].append(low_interval)

        num_to_return = min(n, len(all_harmonic))
        melodic = [voice[-num_to_return:] for voice in all_melodic]
        return all_harmonic[-num_to_return:], melodic

    def interval_or_none(self, first, second):
        if first is None or second is None:
            return None
        else:
            return NamedInterval.from_pitch_carriers(first, second)
