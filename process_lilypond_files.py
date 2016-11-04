import os
import subprocess
import shlex


def process_lilypond_files(directory: str):
    saved_path = os.getcwd()
    os.chdir(directory)
    directory_contents = os.listdir()
    for filename in directory_contents:
        if ".ly" in filename:
            args = shlex.split("lilypond \"%s\"" % filename)
            output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    os.chdir(saved_path)


process_lilypond_files("results")