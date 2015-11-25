#!/bin/bash

TRAINING=$(dirname $0)/training_testing.py
TRAINING_FILES_DIRECTORY=$1
OUTPUT_DIRECTORY=$2

for FILE in $TRAINING_FILES_DIRECTORY/* ; do
    python $TRAINING --options train_model --output_directory $2 --training_set $FILE
done