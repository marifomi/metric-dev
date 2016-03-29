__author__ = 'MarinaFomicheva'

from src.features.impl.abstract_feature import *
from src.lex_resources import config
import numpy


class ExactMatchTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'exact_match_target')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact match for Meteor on target side")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class ExactMatchReference(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'exact_match_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact match for Meteor on reference side")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class FuzzyMatchTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'fuzzy_match_target')
        AbstractFeature.set_description(self, "Proportion of aligned words with fuzzy match for Meteor on target side")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] != 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class FuzzyMatchReference(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'fuzzy_match_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with fuzzy match for Meteor on reference side")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] != 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class AnyMatchTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'any_match_target')
        AbstractFeature.set_description(self, "Proportion of aligned words for Meteor on target side")

    def run(self, cand, ref):

        if len(cand['tokens']) > 0:
            AbstractFeature.set_value(self, len(cand['alignments'][0]) / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class AnyMatchReference(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'any_match_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words for Meteor on reference side")

    def run(self, cand, ref):

        if len(ref['tokens']) > 0:
            AbstractFeature.set_value(self, len(ref['alignments'][0]) / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class MatchContent(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'match_content')
        AbstractFeature.set_description(self, "Proportion of matching content words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for word in cand['alignments'][1]:
                if word[0] not in config.stopwords and word[0] not in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count / numpy.mean([float(len(cand['alignments'][0])),
                                                                float(len(ref['alignments'][0]))]))
        else:
            AbstractFeature.set_value(self, -1)


class MatchFunction(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'match_function')
        AbstractFeature.set_description(self, "Proportion of aligned function words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0
            for word in cand['alignments'][1]:
                if word[0] in config.stopwords or word[0] in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count / numpy.mean([float(len(cand['alignments'][0])),
                                                                float(len(ref['alignments'][0]))]))
        else:
            AbstractFeature.set_value(self, -1)


class FragPenalty(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'frag_penalty')
        AbstractFeature.set_description(self, "Fragmentation penalty from Meteor")

    def run(self, cand, ref):

        chunckNumber = self.calculateChuncks(cand['alignments'][0])
        fragPenalty = 0.0

        if chunckNumber > 1:
            fragPenalty = float(chunckNumber) / len(cand['alignments'][0])

        AbstractFeature.set_value(self, fragPenalty)

    def calculateChuncks(self, alignments):

        sortedAlignments = sorted(alignments, key=lambda alignment: alignment[0])

        chunks = 0
        previousPair = None

        for pair in sortedAlignments:
            if previousPair == None or previousPair[0] != pair[0] - 1 or previousPair[1] != pair[1] - 1:
                chunks += 1
            previousPair = pair

        return chunks


class CountOOV(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_oov')
        AbstractFeature.set_description(self, "Count out-of-vocabulary words using srilm")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['lang_model_sentence_features'][0])


class CountMismatchedOOV(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_mismatched_oov')
        AbstractFeature.set_description(self, "Count mismatched out-of-vocabulary words using srilm")

    def run(self, cand, ref):

        count= 0
        for oov_word in cand['lang_model_word_features']:
            if oov_word not in cand['alignments'][1]:
                count += 1
        AbstractFeature.set_value(self, count)


class LanguModelPerplexity(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_model_perplexity')
        AbstractFeature.set_description(self, "Language model perplexity using srilm")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['lang_model_sentence_features'][2])


class LanguModelLogProbability(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_model_log_probability')
        AbstractFeature.set_description(self, "Language model log probability")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['lang_model_sentence_features'][1])
