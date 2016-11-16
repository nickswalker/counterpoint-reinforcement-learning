import os
import shlex
import subprocess
import sys


def process_lilypond_files(directory: str):
    saved_path = os.getcwd()
    os.chdir(directory)
    directory_contents = os.listdir()
    for filename in directory_contents:
        if ".ly" in filename:
            args = shlex.split("lilypond \"%s\"" % filename)
            output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
            midi_name = filename.replace(".ly", ".midi")
            wav_name = filename.replace(".ly", ".wav")
            args = shlex.split("timidity \"%s\" -Ow2 -o \"%s\"" % (midi_name, wav_name))
            output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    os.chdir(saved_path)


process_lilypond_files(sys.argv[1])
