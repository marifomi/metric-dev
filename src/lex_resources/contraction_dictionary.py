import os.path

class ContractionDictionary(object):

    contraction_table = {}

    def __init__(self, language):
        with open(os.path.expanduser('~/workspace/upf-cobalt/src/lex_resources/contractions/' + language + '.contractions')) as f:
            for line in f:
                words = line.split(',')
                self.contraction_table[words[0].strip()] = words[1].strip()

    def check_contraction(self, word1, word2):
        if word1 not in self.contraction_table and word2 not in self.contraction_table:
            return False

        if (word1 in self.contraction_table and self.contraction_table[word1] == word2) or \
           (word2 in self.contraction_table and self.contraction_table[word2] == word1):
            return True

        return False