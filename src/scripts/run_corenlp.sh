#!/usr/bin/env bash
# export STANFORD_CORENLP_DIR=/Users/MarinaFomicheva/workspace/stanford-corenlp-python/stanford-corenlp-full-2014-08-27

java -mx5g -cp "$STANFORD_CORENLP_DIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat text -outputDirectory "$1" -props "/Users/MarinaFomicheva/workspace/stanford-corenlp-python/parse.properties" -filelist "/Users/MarinaFomicheva/workspace/stanford-corenlp-python/fileList.txt"
