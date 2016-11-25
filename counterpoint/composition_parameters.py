from typing import Tuple, List

from abjad.tools.durationtools import Duration
from abjad.tools.metertools import Meter
from abjad.tools.pitchtools import NamedPitch
from abjad.tools.pitchtools import PitchRange
from abjad.tools.tonalanalysistools import Scale


class CompositionParameters:
    def __init__(self, desired_voices: List[Tuple[str, PitchRange]], meter: Meter, scale: Scale, duration: Duration):
        assert meter.is_simple
        self.scale = scale
        self.duration = duration
        self.meter = meter
        self.number_of_voices_to_generate = len(desired_voices)
        self.desired_voices = desired_voices

        self.pitch_indices = self._generate_pitch_indices()
        self.num_pitches_per_voice = [len(indices) for indices in self.pitch_indices]

    def _pitch_to_range_index(self, voice_index: int, pitch: NamedPitch) -> int:
        return self.pitch_indices[voice_index][pitch]

    def _generate_pitch_indices(self):
        indices = []
        for _, pitch_range in self.desired_voices:
            pitch_set = self.scale.create_named_pitch_set_in_pitch_range(pitch_range)
            pitch_to_index = {}
            for i, pitch in zip(range(0, len(pitch_set)), pitch_set):
                pitch_to_index[pitch] = i
            indices.append(pitch_to_index)
        return indices
