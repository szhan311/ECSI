#!/bin/bash

# Define the variables
GPU=1
SESSION_NAME="diode64"
PRED_MODE="linear1"
SUFFIX=""
CONFIG_DIR="./configs/config_diode64.json"
SMOOTH=0

# Start a detached screen session with the specified name
screen -dmS $SESSION_NAME

# Attach to the screen session
screen -r $SESSION_NAME -X stuff $'\n'

# Activate the conda environment
screen -r $SESSION_NAME -X stuff 'conda activate sibm'$(echo -ne '\015')

# Run the training script with specified configuration
screen -r $SESSION_NAME -X stuff "CUDA_VISIBLE_DEVICES=\"$GPU\" python train.py --config-dir \"$CONFIG_DIR\" --pred-mode \"$PRED_MODE\" --suffix \"$SUFFIX\" --smooth $SMOOTH"$(echo -ne '\015')
# screen -r $SESSION_NAME -X stuff "CUDA_VISIBLE_DEVICES=\"$GPU\" kernprof -l -v train.py --config-dir \"$CONFIG_DIR\" --pred-mode \"$PRED_MODE\" --smooth $SMOOTH"$(echo -ne '\015')

# Attach to the screen session
screen -r $SESSION_NAME