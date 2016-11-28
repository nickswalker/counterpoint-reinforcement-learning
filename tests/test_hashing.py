import unittest

from abjad.tools.durationtools.Duration import Duration
from abjad.tools.pitchtools.NamedPitch import NamedPitch

from counterpoint.composition_environment import CompositionAction, CompositionState
from counterpoint.hashable_note import HashableNote


class TestHashing(unittest.TestCase):
    def test_hashable_note_hashing(self):
        note = HashableNote(NamedPitch("A3"), Duration(1, 4))
        duplicate_note = HashableNote(NamedPitch("A3"), Duration(1, 4))

        self.assertEqual(note, duplicate_note)
        self.assertEqual(hash(note), hash(duplicate_note))

    def test_state_hashing(self):
        note = HashableNote(NamedPitch("A3"), Duration(1, 4))
        duplicate_note = HashableNote(NamedPitch("A3"), Duration(1, 4))
        state = CompositionState(Duration(0), (note, duplicate_note), None)
        duplicate_state = CompositionState(Duration(0), (duplicate_note, note), None)

        self.assertEqual(state, duplicate_state)
        self.assertEqual(hash(state), hash(duplicate_state))

    def test_action_hashing(self):
        note = HashableNote(NamedPitch("A3"), Duration(1, 4))
        duplicate_note = HashableNote(NamedPitch("A3"), Duration(1, 4))
        action = CompositionAction((note, duplicate_note))
        duplicate_action = CompositionAction((duplicate_note, note))

        self.assertEqual(action, duplicate_action)
        self.assertEqual(hash(action), hash(duplicate_action))
