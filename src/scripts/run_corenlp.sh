#!/usr/bin/env bash

java -mx5g -cp "$STANFORD_CORENLP_DIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat text -outputDirectory "$1" -props "/Users/MarinaFomicheva/workspace/stanford-corenlp-python/tok.properties" -filelist "/Users/MarinaFomicheva/workspace/stanford-corenlp-python/fileList.txt"
