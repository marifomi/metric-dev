__author__ = 'MarinaFomicheva'

class Sentence(object):

    def __init__(self):
        self.parse = object
        self.alignments = object
        self.lm = object

    def add_alignments(self, alignments):
        self.alignments = alignments

    def add_lm(self, lm):
        self.lm = lm

    def add_parse(self, parse):
        self.parse = parse

    def add_quest_word(self, quest_word):
        self.quest_word = quest_word
