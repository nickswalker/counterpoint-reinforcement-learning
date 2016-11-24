from abc import abstractmethod

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.scoretools.StaffGroup import StaffGroup
from abjad.tools.topleveltools.iterate import iterate

from counterpoint import constants
from counterpoint.composition_environment import CompositionEnvironment, CompositionState, CompositionAction
from rl.task import Task




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

    def grade_composition(self, composition: CompositionEnvironment) -> int:
        penalties = 0
        working_staff = StaffGroup()
        working_staff.append(composition.voices[0])
        working_staff.append(composition.voices[1])
        for vertical_moment in iterate(working_staff).by_vertical_moment():
            pitches = vertical_moment.leaves
            interval = NamedInterval.from_pitch_carriers(pitches[1], pitches[0])
            if interval.semitones in constants.dissonant_intervals:
                penalties -= 1
            # No intervals greater than a 12th
            if abs(interval.semitones) > NamedInterval("P12").semitones:
                penalties -= 2
            # Tenth is pushing it
            elif abs(interval.semitones) > NamedInterval("M10").semitones:
                penalties -= 1

        return penalties

    def reward(self, state: CompositionState, action: CompositionAction, state_prime: CompositionState):
        return self.grade_composition(self.domain)

    def is_step(self, interval: NamedInterval):
        pass
