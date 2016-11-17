from abc import abstractmethod

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedInterval import NamedInterval

from counterpoint import constants
from counterpoint.composition_environment import CompositionEnvironment, CompositionState, CompositionAction
from counterpoint.music_features import MusicFeatureExtractor
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

        current_upper_pitch = state_prime.voices[0][0].written_pitch
        current_lower_pitch = state_prime.voices[1][0].written_pitch

        harmonic_intervals, melodic_intervals = MusicFeatureExtractor.last_n_intervals(3, self.domain.current_duration,
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

        direction = MusicFeatureExtractor.characterize_relative_motion(contrapuntal_melodic_interval,
                                                                       cantus_melodic_interval)
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
