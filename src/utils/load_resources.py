__author__ = 'MarinaFomicheva'

import re
from src.lex_resources.config import *
import numpy as np


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

def load_word_vectors(vectorsFileName, delimiter=' '):

    vector_file = open(vectorsFileName, 'r')

    for line in vector_file:
        if line == '\n':
            continue

        word = line.strip().split(delimiter)[0]
        vector = np.asfarray(line.strip().split(delimiter)[1:])

        word_vector[word] = vector
