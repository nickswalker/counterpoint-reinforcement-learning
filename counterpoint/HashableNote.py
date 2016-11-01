from abjad.tools.scoretools import Note


class HashableNote(Note):
    def __hash__(self):
        return hash((str(self.written_pitch), self._written_duration))

    def __eq__(self, other):
        if isinstance(other, Note):
            return self.written_pitch == other.written_pitch and self.written_duration == other.written_duration
        return False

    def __le__(self, other):
        return super(HashableNote, self).__le__(other)

    def __ge__(self, other):
        return super(HashableNote, self).__ge__(other)

    def __gt__(self, other):
        return super(HashableNote, self).__gt__(other)

    def __lt__(self, other):
        return super(HashableNote, self).__lt__(other)
