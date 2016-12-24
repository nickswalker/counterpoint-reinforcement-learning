# Counterpoint Reinforcement Learning

An agent for composing species counterpoint.

## Dependencies

Python dependencies are given in `requirements.txt`. TensorFlow must be installed to run the deep learning experiments.
Lillypond, for typesetting compositions. Timidity for MIDI playback and conversion.

## Usage

Execute `runner.py` for a usage message. The experiments in the report were run simply by specifying the task (`0`) and the approach (see the file for the list of these).

Interesting files to look at include `[species_one.py](counterpoint/species/species_one.py)` where you'll find the reward function,
and `[runner.py](runner.py)` where you'll find the list of experiments and the CLI.

## Attribution

[CMAC implementation](https://github.com/stober/cmac) is provided by Jeremy Stober.

Reinforcement learning classes reused from my previous programming assignments.

Substantial musical analysis functionality is provided by [Abjad](http://abjad.mbrsi.org/).