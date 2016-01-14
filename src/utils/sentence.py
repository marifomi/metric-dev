__author__ = 'MarinaFomicheva'

import re
from src.utils.features_reader import FeaturesReader

class Sentence(object):

    def __init__(self):
        self.parse = []
        self.alignments = []
        self.quest_word = []
        self.quest_sent = []
        self.bleu = []
        self.meteor = []

    def add_alignments(self, alignments):
        self.alignments = alignments

    def add_parse(self, parse):
        self.parse = parse

    def add_tokenized(self, tokens):
        self.tokens = tokens

    def add_quest_word(self, quest_word):
        self.quest_word = quest_word

    def add_quest_sent(self, quest_sent):
        self.quest_sent = quest_sent

    def add_bleu(self, bleu):
        self.bleu = bleu

    def add_meteor(self, meteor):
        self.meteor = meteor

