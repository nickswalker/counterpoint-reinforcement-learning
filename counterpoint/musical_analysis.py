from enum import Enum
from typing import Tuple

from abjad.tools.pitchtools.NamedInterval import NamedInterval


def is_step(interval):
    # A major or minor second
    return interval.interval_class == 2 or abs(interval.semitones) < 3


def is_leap(interval):
    # Anything larger than a second
    return interval.interval_class > 2 or abs(interval.semitones) > 3


def same_harmonic_quality(*args):
    first = args[0]
    sameness = True
    for next_elment in args[1:]:
        sameness &= first.interval_class == next_elment.interval_class
    return sameness


def slices_to_melodic_intervals(first, second) -> Tuple[NamedInterval, NamedInterval]:
    first_lower = first.leaves[0]
    second_lower = second.leaves[0]
    first_upper = first.leaves[1]
    second_upper = second.leaves[1]

    lower = NamedInterval.from_pitch_carriers(first_lower, second_lower)
    upper = NamedInterval.from_pitch_carriers(first_upper, second_upper)
    return lower, upper


class RelativeMotion(Enum):
    similar = 0
    parallel = 1
    contrary = 2
    oblique = 3
    none = 4


def slices_to_motion(first, second) -> RelativeMotion:
    lower, upper = slices_to_melodic_intervals(first, second)
    return characterize_relative_motion(upper, lower)


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


class Motion(Enum):
    step_up = 0
    step_down = 1
    leap_up = 2
    leap_down = 3
    none = 4
