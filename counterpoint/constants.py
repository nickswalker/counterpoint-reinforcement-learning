from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.pitchtools.PitchRange import PitchRange

soprano_range = PitchRange("[C4,G5]")
alto_range = PitchRange("[G3, D5]")
tenor_range = PitchRange("[C3, C5]")
bass_range = PitchRange("[E2, C4]")

dissonant_intervals = {NamedInterval("m2").semitones, NamedInterval("m7").semitones,
                       NamedInterval("M7").semitones, NamedInterval("aug2").semitones,
                       NamedInterval("aug4").semitones, NamedInterval("dim5").semitones}
consonant_intervals = {NamedInterval("M2").semitones, NamedInterval("M6").semitones, NamedInterval("M3"),
                       NamedInterval("m3")}
perfect_intervals = {NamedInterval("P4").semitones, NamedInterval("P5").semitones,
                     NamedInterval("P8").semitones, NamedInterval("P11").semitones}
