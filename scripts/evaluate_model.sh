#!/bin/bash

TRAINING=$(dirname $0)/training_testing.py
MODEL_FILES_DIRECTORY=$1
DATA_FILES_DIRECTORY=$2
OUTPUT_DIRECTORY=$3

for MODEL_FILE in $MODEL_FILES_DIRECTORY/* ; do
    for DATA_FILE in $DATA_FILES_DIRECTORY/* ; do
        echo $MODEL_FILE
        echo $DATA_FILE
        python $TRAINING --options evaluate --test_file $DATA_FILE --model_file $MODEL_FILE --output_directory $3
    done
done


