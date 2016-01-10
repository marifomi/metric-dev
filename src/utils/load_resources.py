__author__ = 'MarinaFomicheva'

import re
from src.lex_resources.config import *


def load_ppdb(ppdbFileName):

    global ppdb_dict

    count = 0

    ppdbFile = open(ppdbFileName, 'r')
    for line in ppdbFile:
        if line == '\n':
            continue
        tokens = line.split()
        tokens[1] = tokens[1].strip()
        ppdb_dict[(tokens[0], tokens[1])] = 0.6
        count += 1

def load_word_vectors(vectorsFileName):

    global wordVector
    vectorFile = open (vectorsFileName, 'r')

    for line in vectorFile:
        if line == '\n':
            continue

        match = re.match(r'^([^ ]+) (.+)',line)
        if type(match) is None:
            continue

        word = match.group(1)
        vector = match.group(2)

        word_vector[word] = vector