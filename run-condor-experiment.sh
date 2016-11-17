#!/usr/bin/env bash

###### Arguments ########
# 1) Script to run      #
# 2) Experiment num     #
# 3) Num trials         #
# 4) Output directory   #
# 5) Other script arguments #
#########################

if [ "$#" -ne 5 ]; then
    echo "Incorrect parameters"
    exit 1
fi

# Grab the arguments
SCRIPT_NAME="$1"
EXPERIMENT_NUM="$2"
NUM_TRIALS="$3"
OUTPUT_DIRECTORY="$4"
OTHER_ARGUMENTS="$5"


# Clean the arguments
FULL_SCRIPT_PATH="$( readlink -f $SCRIPT_NAME)"
OUTPUT_DIR="$( readlink -f $OUTPUT_DIRECTORY )"

# Make a timestamped directory in the out dir that was passed in

TIMESTAMP="$(date +%a-%b-%d-%T)"
OUTPUT_DIR="$OUTPUT_DIR/$TIMESTAMP"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/out"


# Form the final arguments string that we'll give to Python
ARGUMENTS="${FULL_SCRIPT_PATH} ${EXPERIMENT_NUM} ${OUTPUT_DIR} -trials 1 -unique-id \$(Process) ${OTHER_ARGUMENTS}"

# Condor wants the full path to the Python executable
PYTHON_PATH="$( which python3.5)"

# Write condor submission file
cat << EOF > submit.condor
Universe        = vanilla
Executable      = $PYTHON_PATH
Error           = $OUTPUT_DIR/logs/err.\$(cluster)
Output          = $OUTPUT_DIR/logs/out.\$(cluster)
Log             = $OUTPUT_DIR/logs/log.\$(cluster)

+Group = "UNDER"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Experiments with applying reinforcement learning to counter point"

Arguments       = $ARGUMENTS
Queue $NUM_TRIALS
EOF

echo "======== Condor Submission ========="
cat submit.condor

condor_submit submit.condor

# Save the submission to the results folder
mv submit.condor $OUTPUT_DIR/arguments.txt