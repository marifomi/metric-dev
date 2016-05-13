from configparser import ConfigParser
import math
from json import *

from src.utils.core_nlp_utils import *
from src.utils import word_sim


class WordInformation(object):

    def __init__(self):
        self.similarity = 0.0
        self.penalty_test = 0.0
        self.penalty_ref = 0.0
        self.penalty_mean = 0.0


class Scorer(object):

    alpha = 1
    beta = 1
    delta = 1

    exact = 1
    stem = 1
    synonym = 1
    paraphrase = 1
    posExact = 1
    posGramCat = 1
    posNone = 1
    related = 1
    related_threshold = 1
    context_importance = 1
    minimal_aligned_relatedness = 1

    arguments = 1
    modifiers = 1
    function = 1

    argument_types = []
    modifier_types = []
    function_types = []

    def __init__(self):
        config = ConfigParser()
        config.readfp(open('config/scorer.cfg'))

        self.alpha = config.getfloat('Scorer', 'alpha')
        self.beta = config.getfloat('Scorer', 'beta')
        self.delta = config.getfloat('Scorer', 'delta')

        self.exact = config.getfloat('Scorer', 'exact')
        self.stem = config.getfloat('Scorer', 'stem')
        self.synonym = config.getfloat('Scorer', 'synonym')
        self.paraphrase = config.getfloat('Scorer', 'paraphrase')
        self.related = config.getfloat('Scorer', 'related')
        self.related_threshold = config.getfloat('Scorer', 'related_threshold')

        self.context_importance = config.getfloat('Scorer', 'context_importance')
        self.minimal_aligned_relatedness = config.getfloat('Scorer', 'minimal_aligned_relatedness')

        self.arguments = config.getfloat('Dependency Weights', 'arguments')
        self.modifiers = config.getfloat('Dependency Weights', 'modifiers')
        self.function = config.getfloat('Dependency Weights', 'function')

        self.argument_types = loads(config.get('Dependency Types', 'arguments'))
        self.modifier_types = loads(config.get('Dependency Types', 'modifiers'))
        self.function_types = loads(config.get('Dependency Types', 'function'))
        self.noisy_types = loads(config.get('Dependency Types', 'noise'))


    def get_dependency_weight(self, dependency_label):

        if dependency_label.split('_')[0] in self.argument_types:
            return self.arguments

        elif dependency_label.split('_')[0] in self.modifier_types:
            return self.modifiers

        else:
            return self.function

    def sum_dependency_weights(self, dependencies):
        result = 0
        for d in dependencies:
            result += self.get_dependency_weight(d)

        return result

    def get_penalties(self, context_info, type):

        source_diff = self.sum_dependency_weights(context_info['srcDiff'])
        target_diff = self.sum_dependency_weights(context_info['tgtDiff'])
        source_length = self.sum_dependency_weights(context_info['srcCon'])
        target_length = self.sum_dependency_weights(context_info['tgtCon'])

        pen = 0.0

        if type == 'test':
            if source_length > 0:
                pen = source_diff/source_length * math.log(source_length + 1.0)
        elif type == 'ref':
            if target_length > 0:
                pen = target_diff/target_length * math.log(target_length + 1.0)
        else:
            if source_length > 0 and target_length > 0:
                pen = self.get_penalty_mean(source_diff/source_length, target_diff/target_length) * math.log(target_length + 1.0)

        return self.normalize_penalty(pen)

    def get_penalty_mean(self, pen_test, pen_ref):

        if pen_ref == 0 or pen_test == 0:
            return max(pen_ref, pen_test)
        else:
            return (1 + math.pow(self.beta, 2)) * (pen_test * pen_ref/((pen_test * math.pow(self.beta, 2)) + pen_ref))

    def normalize_penalty(self, penalty):

        return 2 * (1.0/(1.0 + math.exp(-penalty))) - 1


    def sentence_length(self, sentence):
        return len(prepareSentence2(sentence))

    def word_scores(self, sentence1, sentence2, alignments):

        word_scores = []

        for i, a in enumerate(alignments[0]):
            word_info = WordInformation()
            word_info.similarity = word_sim.word_relatedness_scoring(sentence1[a[0] - 1], sentence2[a[1] - 1], self)
            word_info.penalty_test = self.get_penalties(alignments[2][i], 'test')
            word_info.penalty_ref = self.get_penalties(alignments[2][i], 'ref')
            word_info.penalty_mean = self.get_penalties(alignments[2][i], 'mean')

            word_scores.append(word_info)

        return word_scores

    def sentence_score_cobalt(self, sentence1, sentence2, alignments, word_level_scores):

        functional_words1 = list(filter(lambda x: word_sim.function_word(x.form), sentence1))
        functional_words2 = list(filter(lambda x: word_sim.function_word(x.form), sentence2))

        weighted_length1 = self.delta * (len(sentence1) - len(functional_words1)) + ((1.0 - self.delta) * len(functional_words1))
        weighted_length2 = self.delta * (len(sentence2) - len(functional_words2)) + ((1.0 - self.delta) * len(functional_words2))

        weighted_matches1 = 0
        weighted_matches2 = 0

        for i, a in enumerate(alignments[0]):

            if not word_sim.function_word(sentence1[a[0] - 1].form):
                weighted_matches1 += self.delta * (max(word_level_scores[i].similarity - word_level_scores[i].penalty_mean, self.minimal_aligned_relatedness))
            else:
                weighted_matches1 += (1 - self.delta) * (max(word_level_scores[i].similarity - word_level_scores[i].penalty_mean, self.minimal_aligned_relatedness))

            if not word_sim.function_word(sentence2[a[1] - 1].form):
                weighted_matches2 += self.delta * (max(word_level_scores[i].similarity - word_level_scores[i].penalty_mean, self.minimal_aligned_relatedness))
            else:
                weighted_matches2 += (1 - self.delta) * (max(word_level_scores[i].similarity - word_level_scores[i].penalty_mean, self.minimal_aligned_relatedness))

        if weighted_length1 == 0:
            precision = weighted_matches1
        else:
            precision = weighted_matches1 / weighted_length1

        if weighted_length2 == 0:
            recall = weighted_matches2
        else:
            recall = weighted_matches2 / weighted_length2

        if precision == 0 or recall == 0 or (((1.0 - self.alpha) / precision) + (self.alpha / recall)) == 0:
            fmean = 0
        else:
            fmean = 1.0 / (((1.0 - self.alpha) / precision) + (self.alpha / recall))

        score = fmean

        return score