from collections import defaultdict


class Sentence(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def add_data(self, method, sent_data):
        if 'aligner' in method:
            self['alignments'] = sent_data
        elif 'tokenizer' in method:
            self['tokens'] = sent_data
        else:
            self[method] = sent_data


