#!/bin/bash

# Define the variables
SESSION_NAME="sample_e2h"
GPU=3
PYTHON_FILE_NAME="sample_e2h.py"

# Start a detached screen session with the specified name
screen -dmS $SESSION_NAME

# Attach to the screen session
screen -r $SESSION_NAME -X stuff $'\n'

# Activate the conda environment
screen -r $SESSION_NAME -X stuff 'conda activate ecsi'$(echo -ne '\015')

# Run the training script with specified configuration
screen -r $SESSION_NAME -X stuff "CUDA_VISIBLE_DEVICES=\"$GPU\" python $PYTHON_FILE_NAME"$(echo -ne '\015')
# screen -r $SESSION_NAME -X stuff "CUDA_VISIBLE_DEVICES=\"$GPU\" kernprof -l -v $PYTHON_FILE_NAME"$(echo -ne '\015')

# Attach to the screen session
screen -r $SESSION_NAME