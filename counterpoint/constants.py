from abjad.tools.pitchtools.NamedInterval import NamedInterval
from abjad.tools.pitchtools.PitchRange import PitchRange

soprano_range = PitchRange("[C4,G5]")
alto_range = PitchRange("[G3, D5]")
tenor_range = PitchRange("[C2, G4]")
bass_range = PitchRange("[E3, C4]")

dissonant_intervals = {NamedInterval("m2"), NamedInterval("m7"), NamedInterval("M7"), NamedInterval("aug2"),
                       NamedInterval("aug4"), NamedInterval("dim5")}
consonant_intervals = {NamedInterval("M2"), NamedInterval("M6")}
perfect_intervals = {NamedInterval("P4"), NamedInterval("P5"), NamedInterval("P8"), NamedInterval("P11")}
