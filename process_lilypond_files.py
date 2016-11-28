import os
import shlex
import shutil
import subprocess
import sys


def prepare_directory(directory: str):
    midi_dir = os.path.join(directory, "midi")
    pdf_dir = os.path.join(directory, "pdf")
    wav_dir = os.path.join(directory, "wav")

    shutil.rmtree(midi_dir, ignore_errors=True, onerror=None)
    shutil.rmtree(pdf_dir, ignore_errors=True, onerror=None)
    shutil.rmtree(wav_dir, ignore_errors=True, onerror=None)
    os.makedirs(midi_dir)
    os.makedirs(pdf_dir)
    os.makedirs(wav_dir)

def process_lilypond_files(directory: str):
    saved_path = os.getcwd()
    prepare_directory(directory)

    os.chdir(directory)
    directory_contents = os.listdir()
    for filename in directory_contents:
        if ".ly" in filename:
            args = shlex.split("lilypond \"%s\"" % filename)
            output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
            pdf_name = filename.replace(".ly", ".pdf")
            midi_name = filename.replace(".ly", ".midi")
            wav_name = filename.replace(".ly", ".wav")
            args = shlex.split("timidity \"%s\" -Ow2 -o \"%s\"" % (midi_name, wav_name))
            output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
            os.rename(midi_name, os.path.join("midi", midi_name))
            os.rename(wav_name, os.path.join("wav", wav_name))
            os.rename(pdf_name, os.path.join("pdf", pdf_name))

    os.chdir(saved_path)


process_lilypond_files(sys.argv[1])
