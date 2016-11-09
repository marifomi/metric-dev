from lex_resources.config import *

class Word(object):

    index = -1

    form = None

    lemma = None

    pos = None

    ner = None

    dep = ''

    collapsed = False

    children = []

    parents = []

    category = None

    head = None

    stopword = None

    punctuation = None

    def __init__(self, index, form):
        self.index = index
        self.form = form

    def find_children_nodes(self, sentence_parse):
        if len(self.children) == 0:
            for i, word in enumerate(sentence_parse):
                if word.head is not None and word.head.index == self.index:
                    self.children.append(word)

        return self.children

    def set_children_nodes(self, nodes):
        if len(self.children) == 0:
            self.children = nodes

    def get_category(self):
        pos_codes = {'n': 'noun', 'j': 'adjective', 'v': 'verb', 'r': 'adverb'}
        if self.category is None:
            if self.pos[0] in pos_codes.keys():
                self.category = pos_codes[self.pos[0]]
            else:
                self.category = ''
        return self.category

    def is_sentence_ending_punctuation(self):
        return self.form in ['.', '?', '!']

    def is_named_entity(self):
        return self.ner is not None and self.ner != 'O'

    def is_stopword(self):
        if self.stopword is None:
            global cobalt_stopwords
            self.stopword = (self.lemma in cobalt_stopwords + ['\'s', '\'d', '\'ll'])

        return self.stopword

    def is_punctuation(self):
        if self.punctuation is None:
            global punctuations
            self.punctuation = (self.lemma in punctuations)

        return self.punctuation

    def is_content_word(self):
        return not self.is_stopword() and not self.is_punctuation()
