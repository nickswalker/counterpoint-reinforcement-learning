from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.pitchtools.PitchRange import PitchRange

soprano_range = PitchRange("[C4,G5]")
alto_range = PitchRange("[G3, D5]")
tenor_range = PitchRange("[C2, G4]")
bass_range = PitchRange("[E3, C4]")

dissonant_intervals = {NamedInterval("m2").interval_class, NamedInterval("m7").interval_class,
                       NamedInterval("M7").interval_class, NamedInterval("aug2").interval_class,
                       NamedInterval("aug4").interval_class, NamedInterval("dim5").interval_class}
consonant_intervals = {NamedInterval("M2").interval_class, NamedInterval("M6").interval_class}
perfect_intervals = {NamedInterval("P4").interval_class, NamedInterval("P5").interval_class,
                     NamedInterval("P8").interval_class, NamedInterval("P11").interval_class}
