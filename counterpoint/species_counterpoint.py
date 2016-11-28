from abc import abstractmethod
from typing import Tuple

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.scoretools.StaffGroup import StaffGroup
from abjad.tools.tonalanalysistools.ScaleDegree import ScaleDegree
from abjad.tools.topleveltools.iterate import iterate

from counterpoint import constants
from counterpoint.composition_environment import CompositionEnvironment, CompositionState, CompositionAction
from counterpoint.music_features import MusicFeatureExtractor, RelativeMotion
from rl.task import Task


class CounterpointTask(Task):
    def __init__(self, domain: CompositionEnvironment):
        super().__init__(domain)
        self.domain = domain

    def stateisfinal(self, state: CompositionState):
        if self.domain.composition_parameters.duration <= state.preceding_duration:
            return True
        return False

    @abstractmethod
    def reward(self, state, action, state_prime):
        return 0


class SpeciesOneCounterpoint(CounterpointTask):
    def __init__(self, domain: CompositionEnvironment):
        super().__init__(domain)
        self.prev_duration = Duration(0)
        self.prev_grade = 0.0

    def grade_composition(self, composition: CompositionEnvironment) -> int:
        penalties = 0
        working_staff = StaffGroup()
        working_staff.append(composition.voices[0])
        working_staff.append(composition.voices[1])

        interval_tally = {}
        pitch_tally = {}
        melodic_interval_tally = {}
        vertical_moments = list(iterate(working_staff).by_vertical_moment())
        intervals = []
        degrees = [[], []]

        for moment in vertical_moments:
            pitches = moment.leaves
            interval = NamedInterval.from_pitch_carriers(pitches[1], pitches[0])
            intervals.append(interval)
            degrees[0].append(composition.composition_parameters.scale.named_pitch_class_to_scale_degree(pitches[0]))
            degrees[1].append(composition.composition_parameters.scale.named_pitch_class_to_scale_degree(pitches[1]))

            count = pitch_tally.setdefault(pitches[0], 0) + 1
            pitch_tally[pitches[0]] = count

            count = pitch_tally.setdefault(pitches[1], 0) + 1
            pitch_tally[pitches[1]] = count

            count = interval_tally.setdefault(interval, 0) + 1
            interval_tally[interval] = count

        for i in range(len(vertical_moments)):
            vertical_moment = vertical_moments[i]
            interval = intervals[i]

            # No voice crossing
            if interval.direction_string == "down":
                penalties -= 1
            if abs(interval.semitones) in constants.dissonant_intervals:
                penalties -= 1
            # No harmonic intervals greater than a 12th
            if abs(interval.semitones) > NamedInterval("P12").semitones:
                penalties -= 2
            # Tenth is pushing it
            elif abs(interval.semitones) > NamedInterval("M10").semitones:
                penalties -= 1

            maximum_extent = vertical_moment.offset + vertical_moment.leaves[0].written_duration
            if maximum_extent == composition.composition_parameters.duration:

                prev_top = degrees[0][-2]
                prev_bottom = degrees[1][-2]
                if degrees[0][-1] != ScaleDegree(1) or degrees[1][-1] != ScaleDegree(1):
                    penalties -= 1
                if not (prev_top == ScaleDegree(7) and prev_bottom == ScaleDegree(2) or (
                        prev_top == ScaleDegree(2) and prev_bottom == ScaleDegree(7))):
                    penalties -= 1
                # Make sure we approach the end with contrary stepwise motion
                prev_slice = vertical_moments[i - 1]
                motion = self.sices_to_motion(prev_slice, vertical_moment)
                if motion is not RelativeMotion.contrary:
                    penalties -= 1

            elif i is 0:
                # First interval should have a do on the bottom
                if degrees[0][0] != ScaleDegree(1) or degrees[1][0] != ScaleDegree(1):
                    penalties -= 1
                if interval is not NamedInterval("P8"):
                    penalties -= 1
            if i > 0:
                prev_interval = intervals[i - 1]
                prev_slice = vertical_moments[i - 1]

                lower, upper = self.slices_to_melodic_intervals(prev_slice, vertical_moment)

                count = melodic_interval_tally.get(lower, 0) + 1
                melodic_interval_tally[lower] = count

                count = melodic_interval_tally.get(upper, 0) + 1
                melodic_interval_tally[upper] = count

                if abs(lower.semitones) > 4:
                    penalties -= 10
                if abs(upper.semitones) > 4:
                    penalties -= 10

                motion = MusicFeatureExtractor.characterize_relative_motion(upper, lower)
                if motion is RelativeMotion.none:
                    penalties -= 1

                # Never have two perfect consonances in a row
                if prev_interval.semitones == interval.semitones and interval.quality_string == "perfect":
                    penalties -= 1

            if i > 2:
                prev_interval = intervals[i - 1]
                prev_prev_interval = intervals[i - 2]
                prev_prev_prev_interval = intervals[i - 3]

                all_same = interval == prev_interval == prev_prev_interval == prev_prev_prev_interval
                if all_same:
                    if interval.semitones in constants.consonant_intervals:
                        penalties -= 1

        for pitch, num in pitch_tally.items():
            if num > 2:
                penalties -= 5 * num

        for interval, num in interval_tally.items():
            if num > 5:
                penalties -= 5 * num

        for interval, num in melodic_interval_tally.items():
            if num > 5:
                penalties -= 5 * num

        return penalties

    def slices_to_melodic_intervals(self, first, second) -> Tuple[NamedInterval, NamedInterval]:
        first_lower = first.leaves[0]
        second_lower = second.leaves[0]
        first_upper = first.leaves[1]
        second_upper = second.leaves[1]

        lower = NamedInterval.from_pitch_carriers(first_lower, second_lower)
        upper = NamedInterval.from_pitch_carriers(first_upper, second_upper)
        return lower, upper

    def sices_to_motion(self, first, second) -> RelativeMotion:
        lower, upper = self.slices_to_melodic_intervals(first, second)
        return MusicFeatureExtractor.characterize_relative_motion(upper, lower)

    def reward(self, state: CompositionState, action: CompositionAction, state_prime: CompositionState):
        if self.prev_duration == state_prime.preceding_duration:
            assert False
            return
        elif self.prev_duration < state_prime.preceding_duration:
            new_grade = self.grade_composition(self.domain)
            reward = new_grade - self.prev_grade
            self.prev_grade = new_grade
        elif self.prev_duration > state_prime.preceding_duration:
            # This should only be the first step
            reward = self.grade_composition(self.domain)
            self.prev_grade = reward
        self.prev_duration = state_prime.preceding_duration
        return reward

    def is_step(self, interval: NamedInterval):
        pass
