from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.pitchtools.NamedPitch import NamedPitch

from counterpoint.music_features import MusicFeatureExtractor
from counterpoint.species_counterpoint import CounterpointTask


class ThirdsAreGoodTask(CounterpointTask):
    def reward(self, state, action, state_prime):
        harmonic_intervals, melodic_intervals = MusicFeatureExtractor.last_n_intervals(1, self.domain.current_duration,
                                                                                       self.domain.voices[0],
                                                                                       self.domain.given_voices[0])
        if len(melodic_intervals[0]) > 0:
            last_interval = melodic_intervals[0][0]
            if last_interval.interval_class in {NamedInterval("M3").interval_class, NamedInterval("m3").interval_class}:
                return 10
            return -1
        return -1


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
