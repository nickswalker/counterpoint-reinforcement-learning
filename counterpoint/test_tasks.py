from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.pitchtools.NamedPitch import NamedPitch
from abjad.tools.tonalanalysistools.ScaleDegree import ScaleDegree

from counterpoint.composition_environment import CompositionEnvironment
from counterpoint.music_features import MusicFeatureExtractor
from counterpoint.species_counterpoint import CounterpointTask


class ScalesAreGood(CounterpointTask):
    def __init__(self, domain: CompositionEnvironment):
        super().__init__(domain)

    def reward(self, state, action, state_prime):
        i = len(self.domain.voices[0])
        upper = self.domain.composition_parameters.scale.named_pitch_class_to_scale_degree(self.domain.voices[0][-1])
        target = ScaleDegree((i % 8) + 1)
        return -1 * abs(upper.number - target.number)


class ThirdsAreGoodTask(CounterpointTask):
    def reward(self, state, action, state_prime):
        if NamedInterval.from_pitch_carriers(self.domain.voices[1][-1], self.domain.voices[0][-1]).semitones != 4:
            return -1
        return 0


class UnisonsAreGoodTask(CounterpointTask):
    def reward(self, state, action, state_prime):
        harmonic_intervals, melodic_intervals = MusicFeatureExtractor.last_n_intervals(1, self.domain.current_duration,
                                                                                       self.domain.voices[0],
                                                                                       self.domain.given_voices[0])

        if len(harmonic_intervals) > 0:
            last_interval = harmonic_intervals[0]
            if last_interval.interval_class in {NamedInterval("P1").interval_class, NamedInterval("P8").interval_class}:
                return 10
            return -1
        return -1


class OnePitchIsGood(CounterpointTask):
    def reward(self, state, action, state_prime):
        if action.notes_per_voice[0].written_pitch == NamedPitch("A4"):
            return 10
        return -1
