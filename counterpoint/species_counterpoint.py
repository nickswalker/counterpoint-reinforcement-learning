from abc import abstractmethod

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedInterval import NamedInterval

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

    def reward(self, state: CompositionState, action: CompositionAction, state_prime: CompositionState):
        current_reward = 0
        harmonic_interval = NamedInterval.from_pitch_carriers(state_prime.voices[1][0], state_prime.voices[0][0])
        # Is this the first beat?
        if state.preceding_duration == Duration(0):
            if harmonic_interval not in constants.perfect_intervals:
                current_reward -= 1
            return current_reward
        contrapuntal_melodic_interval = NamedInterval.from_pitch_carriers(state.voices[0][0], action.notes_per_voice[0])
        cantus_melodic_interval = NamedInterval.from_pitch_carriers(state.voices[1][0], state.voices[1][1])

        parallel_motion = contrapuntal_melodic_interval.direction_number == cantus_melodic_interval.direction_number and contrapuntal_melodic_interval == cantus_melodic_interval

        if harmonic_interval in constants.dissonant_intervals:
            current_reward -= 1
        if parallel_motion:
            current_reward -= 1

        return current_reward
