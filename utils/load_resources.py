import os
import numpy as np

from lex_resources.config import *


def load_ppdb(ppdbFileName):

    global ppdb_dict

    count = 0

    ppdbFile = open(os.path.expanduser(ppdbFileName), 'r')
    for line in ppdbFile:
        if line == '\n':
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        tokens[1] = tokens[1].strip()
        ppdb_dict[(tokens[0], tokens[1])] = 0.6
        count += 1


def load_word_vectors(vectorsFileName, delimiter=' '):

    vector_file = open(os.path.expanduser(vectorsFileName), 'r')

    for line in vector_file:
        if line == '\n':
            continue

        word = line.strip().split(delimiter)[0]
        vector = np.asfarray(line.strip().split(delimiter)[1:])

        word_vector[word] = vector
