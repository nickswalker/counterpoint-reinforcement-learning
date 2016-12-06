from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.scoretools.StaffGroup import StaffGroup
from abjad.tools.tonalanalysistools.ScaleDegree import ScaleDegree
from abjad.tools.topleveltools.iterate import iterate

from counterpoint import constants
from counterpoint.composition_environment import CompositionEnvironment
from counterpoint.musical_analysis import is_step, is_leap, same_harmonic_quality, slices_to_melodic_intervals, \
    characterize_relative_motion, RelativeMotion
from counterpoint.species_counterpoint import GradedCounterpointTask


class SpeciesOneCounterpoint(GradedCounterpointTask):
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
            harmonic = NamedInterval.from_pitch_carriers(pitches[1], pitches[0])
            intervals.append(harmonic)
            degrees[0].append(composition.composition_parameters.scale.named_pitch_class_to_scale_degree(pitches[0]))
            degrees[1].append(composition.composition_parameters.scale.named_pitch_class_to_scale_degree(pitches[1]))

            count = pitch_tally.setdefault(pitches[0], 0) + 1
            pitch_tally[pitches[0]] = count

            count = pitch_tally.setdefault(pitches[1], 0) + 1
            pitch_tally[pitches[1]] = count

            count = interval_tally.setdefault(harmonic, 0) + 1
            interval_tally[harmonic] = count

        for i in range(len(vertical_moments)):
            vertical_moment = vertical_moments[i]
            harmonic = intervals[i]

            # No voice crossing
            if harmonic.direction_string == "descending":
                penalties -= 1
            if abs(harmonic.semitones) in constants.dissonant_intervals:
                penalties -= 5
            # No harmonic intervals greater than a 12th
            if abs(harmonic.semitones) > NamedInterval("P12").semitones:
                penalties -= 5
            # Tenth is pushing it
            elif abs(harmonic.semitones) > NamedInterval("M10").semitones:
                penalties -= 1

            maximum_extent = vertical_moment.offset + vertical_moment.leaves[0].written_duration
            if maximum_extent == composition.composition_parameters.duration:
                last_top = degrees[0][-1]
                last_bottom = degrees[1][-1]
                prev_top = degrees[0][-2]
                prev_bottom = degrees[1][-2]
                # Use an authentic cadence
                if last_top != ScaleDegree(1):
                    penalties -= 10
                if last_bottom != ScaleDegree(1):
                    penalties -= 10
                if not ((prev_top == ScaleDegree(7) and prev_bottom == ScaleDegree(2)) or (
                                prev_top == ScaleDegree(2) and prev_bottom == ScaleDegree(7))):
                    penalties -= 10

            elif i is 0:
                # First interval should be a tonic unison
                if degrees[0][0] != ScaleDegree(1):
                    penalties -= 5
                if degrees[1][0] != ScaleDegree(1):
                    penalties -= 5
            if i > 0:
                prev_harmonic = intervals[i - 1]
                prev_slice = vertical_moments[i - 1]

                lower_melodic, upper_melodic = slices_to_melodic_intervals(prev_slice, vertical_moment)

                count = melodic_interval_tally.get(lower_melodic, 0) + 1
                melodic_interval_tally[lower_melodic] = count

                count = melodic_interval_tally.get(upper_melodic, 0) + 1
                melodic_interval_tally[upper_melodic] = count

                motion = characterize_relative_motion(upper_melodic, lower_melodic)
                # Lines should always be moving
                if motion is RelativeMotion.none:
                    penalties -= 5

                # Contrary motion is preferred
                if motion is RelativeMotion.similar or motion is RelativeMotion.oblique:
                    penalties -= 1

                # Never have two perfect consonances in a row
                if prev_harmonic.interval_class == harmonic.interval_class and harmonic.quality_string == "perfect":
                    penalties -= 5

                # Steps are preferred to leaps
                if is_leap(lower_melodic):
                    penalties -= 1

                if is_leap(upper_melodic):
                    penalties -= 1
                if i > 2:
                    prev_prev_harmonic = intervals[i - 2]
                    prev_prev_prev_harmonic = intervals[i - 3]

                    all_same = same_harmonic_quality(harmonic, prev_harmonic, prev_prev_harmonic,
                                                     prev_prev_prev_harmonic)
                    # Don't have the same interval more than three times in a row
                    if all_same:
                        penalties -= 5

                    prev_lower_melodic, prev_upper_harmonic = slices_to_melodic_intervals(vertical_moments[i - 2],
                                                                                          prev_slice)

                    # Encourage counterstepwise motion.
                    # If the prev motion was a leap..
                    if is_leap(prev_lower_melodic):
                        # It needs to be resolved by a step, and the step needs to be in the opposite direction
                        if not is_step(lower_melodic) or (
                                        prev_lower_melodic.semitones > 0 ^ lower_melodic.semitones > 0):
                            penalties -= 5
                        else:
                            ()
                            # print("we did it")

                    if is_leap(prev_upper_harmonic):
                        if not is_step(upper_melodic) or (
                                        prev_upper_harmonic.semitones > 0 ^ upper_melodic.semitones > 0):
                            penalties -= 5
                        else:
                            ()
                            # print("we did it")

        """
        for pitch, num in pitch_tally.items():
            if num > 2:
                penalties -= 5 * num

        for interval, num in interval_tally.items():
            if num > 5:
                penalties -= 5 * num

        for interval, num in melodic_interval_tally.items():
            if num > 5:
                penalties -= 5 * num
        """

        return penalties
