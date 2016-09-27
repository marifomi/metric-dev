import codecs

from utils.word import Word

__author__ = 'marina'


class CONNL(object):

    @staticmethod
    def load(path_input):
        sentences = []
        sentence = []
        with codecs.open(path_input,'r', 'utf8') as f:
            lines = f.readlines()

        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                sentence = []
                continue

            parts = line.strip().split('\t')
            word = Word(parts[0], parts[1], parts[2], parts[4], parts[7]) # punctuation head is root
            word.head = parts[6]
            sentence.append(word)

        sentences.append(sentence)
        return sentences
