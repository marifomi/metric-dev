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
            word = Word(parts[0], parts[1]) # punctuation head is root
            word.lemma = parts[2]
            word.pos = parts[4]
            word.dep = parts[7]
            word.head = parts[6]
            sentence.append(word)

        sentences.append(sentence)
        return sentences
