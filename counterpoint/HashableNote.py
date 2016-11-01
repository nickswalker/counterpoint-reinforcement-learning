from abjad.tools.scoretools import Note


class HashableNote(Note):
    def __hash__(self):
        return hash((str(self.written_pitch), self._written_duration))

    def __eq__(self, other):
        if isinstance(other, Note):
            return self.written_pitch == other.written_pitch and self.written_duration == other.written_duration
        return False
