from abjad.tools.indicatortools.KeySignature import KeySignature
from abjad.tools.metertools.Meter import Meter
from abjad.tools.scoretools.Voice import Voice

cantus_firmi = [
    (Voice("a4 e fs e a b cs fs gs b a2", name="cantus"), Meter(4, 4), KeySignature("a", "major"))
]
