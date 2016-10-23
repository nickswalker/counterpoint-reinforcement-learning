from abjad.tools.metertools.Meter import Meter
from abjad.tools.scoretools.Voice import Voice
from abjad.tools.tonalanalysistools.Scale import Scale

cantus_firmi = [
    (Voice("a4 e fs e a b cs fs gs b a2", name="cantus"), Meter(4, 4), Scale("a", "major"))
]
