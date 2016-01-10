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

    def add_quest_word(self, quest_word):
        self.quest_word = quest_word

    def add_quest_sent(self, quest_sent):
        self.quest_sent = quest_sent

    def add_bleu(self, scores_file):

        for line in open(scores_file).readlines():
            if not line.startswith('  BLEU'):
                continue
            self.bleu.append(float(re.sub(r'^.+= ([01]\.[0-9]+).+$', r'\1', line.strip())))

    def add_meteor(self, scores_file):

        for line in open(scores_file).readlines():
            if not line.startswith('Segment '):
                continue
            self.meteor.append(float(line.strip().split('\t')[1]))

