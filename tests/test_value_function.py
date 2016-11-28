import unittest

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedPitch import NamedPitch

from counterpoint.composition_environment import CompositionAction, CompositionState
from counterpoint.hashable_note import HashableNote
from rl.valuefunction.tabular import StateActionValueTable


class TestValueFunction(unittest.TestCase):
    def setUp(self):
        self.allowed_note = HashableNote(NamedPitch("A3"), Duration(1, 4))
        self.allowed_action = CompositionAction((self.allowed_note, self.allowed_note))
        self.value_function = StateActionValueTable(actions=[self.allowed_action])

    def test_state_value_function(self):
        state = CompositionState(Duration(0), (self.allowed_note, None), None)
        state_two = CompositionState(Duration(0), (self.allowed_note, self.allowed_note), None)
        self.assertEqual(0.0, self.value_function.actionvalue(state, self.allowed_action))
        self.value_function.setactionvalue(state, self.allowed_action, 1.0)
        self.assertEqual(1.0, self.value_function.actionvalue(state, self.allowed_action))

        self.assertEqual(0.0, self.value_function.actionvalue(state_two, self.allowed_action))
