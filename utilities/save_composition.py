import os

from abjad.tools import lilypondfiletools
from abjad.tools import markuptools
from abjad.tools import scoretools
from abjad.tools.durationtools.Duration import Duration
from abjad.tools.indicatortools.Clef import Clef
from abjad.tools.indicatortools.Tempo import Tempo
from abjad.tools.scoretools.Score import Score
from abjad.tools.scoretools.Staff import Staff
from abjad.tools.topleveltools.attach import attach

from counterpoint.composition_environment import CompositionEnvironment


def save_composition(name: str, agent_name: str, composition: CompositionEnvironment, out_dir: str):
    score = Score()
    staff_group = scoretools.StaffGroup([], context_name='StaffGroup')

    for voice in composition.voices + composition.given_voices:
        staff = Staff([voice])
        if voice.name == "cantus":
            attach(Clef("bass"), staff)
        attach(composition.scale.key_signature, staff)

        staff_group.append(staff)
    tempo = Tempo(Duration(1, 4), 100)
    attach(tempo, staff_group[0])
    score.append(staff_group)
    score.add_final_bar_line()

    lilypond_file = lilypondfiletools.make_basic_lilypond_file(score)
    lilypond_file.header_block.composer = markuptools.Markup(agent_name)
    lilypond_file.header_block.title = markuptools.Markup(name)
    lilypond_file.header_block.tagline = markuptools.Markup()

    midi_block = lilypondfiletools.Block(name="midi")
    context_block = lilypondfiletools.Block(name="context")
    channel_mapping = lilypondfiletools.Block(name="score")

    # channel_mapping.midiChannelMapping = "instrument"
    # context_block.items.append(channel_mapping)
    # midi_block.items.append(context_block)

    lilypond_file.score_block.items.append(midi_block)
    layout_block = lilypondfiletools.Block(name="layout")
    lilypond_file.score_block.items.append(layout_block)

    filename = name + ".ly"
    prefix = "results/" + out_dir + "/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    with open(prefix + filename, mode="w") as f:
        f.write(format(lilypond_file))
