__author__ = 'u88591'

from src.scorer.scorer import Scorer
from src.lex_resources import config
import numpy
from src.utils import word_sim
from src.features.impl.abstract_feature import *
from gensim import matutils
from numpy import array, dot
import math
from scipy.spatial import distance
from src.utils.clean_punctuation import CleanPunctuation

class CountWordsTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_words_target')
        AbstractFeature.set_description(self, "Number of words in the candidate")

    def run(self, cand, ref):
         AbstractFeature.set_value(self, len(cand['tokens']))


class CountWordsRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_words_ref')
        AbstractFeature.set_description(self, "Number of words in the reference")

    def run(self, cand, ref):
         AbstractFeature.set_value(self, len(ref['tokens']))


class LengthsRatio(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lengths_ratio')
        AbstractFeature.set_description(self, "Ratio of candidate and reference sentence lengths")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['tokens'])/float(len(ref['tokens'])))


class CountContentTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_content_target')
        AbstractFeature.set_description(self, "Number of content words in the candidate")

    def run(self, cand, ref):
        count = 0

        for word in cand['tokens']:
            if not word_sim.functionWord(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountContentRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_content_ref')
        AbstractFeature.set_description(self, "Number of content words in the reference")

    def run(self, cand, ref):
        count = 0

        for word in ref['tokens']:
             if not word_sim.functionWord(word):
                 count += 1

        AbstractFeature.set_value(self, count)


class CountFunctionTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_function_target')
        AbstractFeature.set_description(self, "Number of function words in the candidate")

    def run(self, cand, ref):
        count = 0

        for word in cand['tokens']:
            if word_sim.functionWord(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountFunctionRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_function_ref')
        AbstractFeature.set_description(self, "Number of function words in the reference")

    def run(self, cand, ref):
        count = 0

        for word in ref['tokens']:
            if word_sim.functionWord(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountAligned(AbstractFeature):

    def __init__(self):
       AbstractFeature.__init__(self)
       AbstractFeature.set_name(self, 'count_aligned')
       AbstractFeature.set_description(self, "Number of aligned words")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['alignments'][0]))


class CountNonAlignedTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_target')
        AbstractFeature.set_description(self, "Number of non-aligned words")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['tokens']) - len(cand['alignments'][0]))


class CountNonAlignedRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_ref')
        AbstractFeature.set_description(self, "Number of non-aligned words")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(ref['tokens']) - len(ref['alignments'][0]))


class PropNonAlignedTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_target')
        AbstractFeature.set_description(self, "Proportion of non-aligned words in the candidate")

    def run(self, cand, ref):

        if len(cand['tokens']) > 0:
            AbstractFeature.set_value(self, len(cand['tokens']) - len(cand['alignments'][0]) / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class PropNonAlignedRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_ref')
        AbstractFeature.set_description(self, "Proportion of non-aligned words in the reference")

    def run(self, cand, ref):

        if len(ref['tokens']) > 0:
            AbstractFeature.set_value(self, len(ref['tokens']) - len(cand['alignments'][0]) / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)

class PropAlignedTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_target')
        AbstractFeature.set_description(self, "Proportion of aligned words in the candidate")

    def run(self, cand, ref):

        if len(cand['tokens']) > 0:
            AbstractFeature.set_value(self, len(cand['alignments'][0]) / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_ref')
        AbstractFeature.set_description(self, "Proportion of aligned words in the reference")

    def run(self, cand, ref):

        if len(ref['parse']) > 0:
            AbstractFeature.set_value(self, len(cand['alignments'][0]) / float(len(ref['parse'])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedContent(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_content')
        AbstractFeature.set_description(self, "Count of aligned content words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for word in cand['alignments'][1]:
                if word[0] not in config.stopwords and word[0] not in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count)


class CountAlignedFunction(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_function')
        AbstractFeature.set_description(self, "Count of aligned function words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0
            for word in cand['alignments'][1]:
                if word[0] in config.stopwords or word[0] in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count)


class CountNonAlignedContent(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_content')
        AbstractFeature.set_description(self, "Count of non aligned content words")

    def run(self, cand, ref):

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if word.lower() not in config.stopwords and word.lower() not in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count)


class CountNonAlignedFunction(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_function')
        AbstractFeature.set_description(self, "Count of non-aligned function words")

    def run(self, cand, ref):

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if word.lower() in config.stopwords and word.lower() not in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count)


class PropNonAlignedContent(AbstractFeature):

    # On target side

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_content')
        AbstractFeature.set_description(self, "Proportion of non aligned content words")

    def run(self, cand, ref):

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if word.lower() not in config.stopwords and word.lower() not in config.punctuations:
                    count += 1

            if len(cand['tokens']) == len(cand['alignments'][0]):
                AbstractFeature.set_value(self, 0)
            else:
                AbstractFeature.set_value(self, count/float(len(cand['tokens']) - len(cand['alignments'][0])))


class PropNonAlignedFunction(AbstractFeature):

    # On target side

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_function')
        AbstractFeature.set_description(self, "Prop of non-aligned function words")

    def run(self, cand, ref):

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if word.lower() in config.stopwords and word.lower() not in config.punctuations:
                    count += 1

            if len(cand['tokens']) == len(cand['alignments'][0]):
                AbstractFeature.set_value(self, 0)
            else:
                AbstractFeature.set_value(self, count/float(len(cand['tokens']) - len(cand['alignments'][0])))


class PropAlignedContent(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_content')
        AbstractFeature.set_description(self, "Proportion of aligned content words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for word in cand['alignments'][1]:
                if word[0] not in config.stopwords and word[0] not in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedFunction(AbstractFeature):

    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_function')
        AbstractFeature.set_description(self, "Proportion of aligned function words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0
            for word in cand['alignments'][1]:
                if word[0] in config.stopwords or word[0] in config.punctuations:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedExactExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_exact_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lexical match and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedExactExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_exact_exact')
        AbstractFeature.set_description(self, "Count of aligned words with exact lexical match and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedSynExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_syn_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lexical match and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedSynExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_syn_exact')
        AbstractFeature.set_description(self, "Count of aligned words with synonym lexical match and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedParaExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_para_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase lexical match and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedParaExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_para_exact')
        AbstractFeature.set_description(self, "Count of aligned words with paraphrase lexical match and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedExactCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_exact_coarse')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lexical match and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedExactCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_exact_coarse')
        AbstractFeature.set_description(self, "Count of aligned words with exact lexical match and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedSynCoarse(AbstractFeature):

    def __init__(self):
       AbstractFeature.__init__(self)
       AbstractFeature.set_name(self, 'prop_aligned_syn_coarse')
       AbstractFeature.set_description(self, "Proportion of aligned words with synonym lexical match and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' and word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedSynCoarse(AbstractFeature):

    def __init__(self):
       AbstractFeature.__init__(self)
       AbstractFeature.set_name(self, 'count_aligned_syn_coarse')
       AbstractFeature.set_description(self, "Count of aligned words with synonym lexical match and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' and word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedParaCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_para_coarse')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase lexical match and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedParaCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_para_coarse')
        AbstractFeature.set_description(self, "Count of aligned words with paraphrase lexical match and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase':
                    if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                        count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedSynDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_syn_diff')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lexical match and different POS")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedSynDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_syn_diff')
        AbstractFeature.set_description(self, "Count of aligned words with synonym lexical match and different POS")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedParaDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_para_diff')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase lexical match and different POS")

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedParaDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_para_diff')
        AbstractFeature.set_description(self, "Count of aligned words with paraphrase lexical match and different POS")

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedDistribExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_distrib_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with distributional similarity and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedDistribExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_distrib_exact')
        AbstractFeature.set_description(self, "Count of aligned words with distributional similarity and exact POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional' and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedDistribCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_distrib_coarse')
        AbstractFeature.set_description(self, "Proportion of aligned words with distributional similarity and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' and word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedDistribCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_distrib_coarse')
        AbstractFeature.set_description(self, "Count of aligned words with distributional similarity and coarse POS match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' and word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedDistribDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_distrib_diff')
        AbstractFeature.set_description(self, "Proportion of aligned words with distributional similarity and different POS")

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None' and word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedDistribDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_distrib_diff')
        AbstractFeature.set_description(self, "Count of aligned words with distributional similarity and different POS")

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None' and word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count)


class PropAlignedPosExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_pos_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact pos match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedPosCoarse(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_pos_coarse')
        AbstractFeature.set_description(self, "Proportion of aligned words with coarse pos match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedPosDiff(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_pos_diff')
        AbstractFeature.set_description(self, "Proportion of aligned words with different pos")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexExact(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_lex_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lex match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexSyn(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_lex_syn')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lex match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Synonym':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexPara(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_lex_para')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase lex match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Paraphrase':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexDistrib(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_lex_distrib')
        AbstractFeature.set_description(self, "Proportion of aligned words with distrib lex match")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexExactMeteor(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_exact_meteor')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lex match for Meteor")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexNonExactMeteor(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_non_exact_meteor')
        AbstractFeature.set_description(self, "Proportion of aligned words with non-exact lex match for Meteor")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] != 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexStemMeteor(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_stem_meteor')
        AbstractFeature.set_description(self, "Proportion of aligned words with stem lex match for Meteor")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 1:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexSynMeteor(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_syn_meteor')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lex match for Meteor")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 2:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedLexParaMeteor(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_para_meteor')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase lex match for Meteor")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 3:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class AvgPenExactTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_exact_target')
        AbstractFeature.set_description(self, "Average CP for aligned words with exact match in the candidate (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

                for dep_label in cand['alignments'][2][i]['srcDiff']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        difference += 1

                for dep_label in cand['alignments'][2][i]['srcCon']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        context += 1

                penalty = 0.0
                if context != 0:
                    penalty = difference / context * math.log(context + 1.0)

                if penalty > 0:
                    penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, numpy.mean(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class AvgPenExactRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_exact_ref')
        AbstractFeature.set_description(self, "Average CP for aligned words with exact match in the reference (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

                for dep_label in cand['alignments'][2][i]['tgtDiff']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        difference += 1

                for dep_label in cand['alignments'][2][i]['tgtCon']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        context += 1

                penalty = 0.0
                if context != 0:
                    penalty = difference / context * math.log(context + 1.0)

                if penalty > 0:
                    penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, numpy.mean(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class AvgPenNonExactTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_non_exact_target')
        AbstractFeature.set_description(self, "Average CP for aligned words with non exact match in the candidate (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

                for dep_label in cand['alignments'][2][i]['srcDiff']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        difference += 1

                for dep_label in cand['alignments'][2][i]['srcCon']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        context += 1

                penalty = 0.0
                if context != 0:
                    penalty = difference / context * math.log(context + 1.0)

                if penalty > 0:
                    penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, numpy.mean(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class AvgPenNonExactRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_non_exact_ref')
        AbstractFeature.set_description(self, "Average CP for aligned words with non exact match in the reference (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

                for dep_label in cand['alignments'][2][i]['tgtDiff']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        difference += 1

                for dep_label in cand['alignments'][2][i]['tgtCon']:
                    if not dep_label.split('_')[0] in my_scorer.noisy_types:
                        context += 1

                penalty = 0.0
                if context != 0:
                    penalty = difference / context * math.log(context + 1.0)

                if penalty > 0:
                    penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, numpy.mean(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class PropPenExactTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_exact_target')
        AbstractFeature.set_description(self, "Proportion of words with CP with exact match in the candidate (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

                counter_words += 1

                if len(cand['alignments'][2][i]['srcDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
             AbstractFeature.set_value(self, counter_penalties / counter_words)
        else:
             AbstractFeature.set_value(self, 0.0)


class PropPenExactRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_exact_ref')
        AbstractFeature.set_description(self, "Proportion of words with CP with exact match in the reference (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

                counter_words += 1

                if len(cand['alignments'][2][i]['tgtDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
             AbstractFeature.set_value(self, counter_penalties / counter_words)
        else:
             AbstractFeature.set_value(self, 0.0)


class PropPenNonExactTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_non_exact_target')
        AbstractFeature.set_description(self, "Proportion of words with CP with non exact match in the candidate (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

                counter_words += 1

                if len(cand['alignments'][2][i]['srcDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
             AbstractFeature.set_value(self, counter_penalties / counter_words)
        else:
             AbstractFeature.set_value(self, 0.0)


class PropPenNonExactRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_non_exact_ref')
        AbstractFeature.set_description(self, "Proportion of words with CP with non exact match in the reference (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.wordRelatednessFeature(word_candidate, word_reference) == 'Exact' and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

                counter_words += 1

                if len(cand['alignments'][2][i]['tgtDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
             AbstractFeature.set_value(self, counter_penalties / counter_words)
        else:
             AbstractFeature.set_value(self, 0.0)


class AvgPenTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_target')
        AbstractFeature.set_description(self, "Average CP for the candidate translation (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['srcDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['srcCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, numpy.mean(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class AvgPentRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_ref')
        AbstractFeature.set_description(self, "Average CP for aligned words in the reference translation (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['tgtDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['tgtCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, numpy.mean(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class MinPenTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_pen_target')
        AbstractFeature.set_description(self, "Minimum CP for the candidate translation (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['srcDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['srcCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, min(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class MinPentRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_pen_ref')
        AbstractFeature.set_description(self, "Minimum CP for aligned words in the reference translation (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['tgtDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['tgtCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, min(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class MaxPenTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_pen_target')
        AbstractFeature.set_description(self, "Max CP for the candidate translation (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['srcDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['srcCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, max(penalties))
        else:
            AbstractFeature.set_value(self, 0)


class MaxPentRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_pen_ref')
        AbstractFeature.set_description(self, "Max CP for aligned words in the reference translation (considered only for the words with CP > 0)")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['tgtDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['tgtCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        if len(penalties) > 0:
            AbstractFeature.set_value(self, max(penalties))
        else:
            AbstractFeature.set_value(self, 0)

class PropPen(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen')
        AbstractFeature.set_description(self, "Proportion of words with penalty over the number of aligned words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        counter_penalties = 0.0
        counter_words = 0.0

        for i, index in enumerate(cand['alignments'][0]):

            counter_words += 1

            if len(cand['alignments'][2][i]['srcDiff']) > 0 or len(cand['alignments'][2][i]['tgtDiff']) > 0:
                counter_penalties += 1

        if counter_words > 0:
             AbstractFeature.set_value(self, counter_penalties / counter_words)
        else:
             AbstractFeature.set_value(self, 0.0)


class CountPen(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_pen')
        AbstractFeature.set_description(self, "Count of words with penalty over the number of aligned words")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        counter_penalties = 0.0
        counter_words = 0.0

        for i, index in enumerate(cand['alignments'][0]):

            counter_words += 1

            if len(cand['alignments'][2][i]['srcDiff']) > 0 or len(cand['alignments'][2][i]['tgtDiff']) > 0:
                counter_penalties += 1

        AbstractFeature.set_value(self, counter_penalties)


class PropPenHigh(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_high')
        AbstractFeature.set_description(self, "Prop of words with penalty higher than avg over the number of aligned words")

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, 0)
            return

        for i, index in enumerate(cand['alignments'][0]):

            for dep_label in cand['alignments'][2][i]['tgtDiff']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    difference += 1

            for dep_label in cand['alignments'][2][i]['tgtCon']:
                if not dep_label.split('_')[0] in my_scorer.noisy_types:
                    context += 1

            penalty = 0.0
            if context != 0:
                penalty = difference / context * math.log(context + 1.0)

            if penalty > 0:
                penalties.append(penalty)

        avg_pen = numpy.mean(penalties)
        count_high = 0

        for pen in penalties:
            if pen > avg_pen:
                count_high += 1

        AbstractFeature.set_value(self, count_high / float(len(cand['tokens'])))


class ContextMatch(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'context_match')
        AbstractFeature.set_description(self, "Number of cases with non-aligned function words in exactly matching contexts")

    def run(self, cand, ref):

        align_dict = {}

        for wpair in cand['alignments'][0]:
            align_dict[wpair[0] - 1] = wpair[1] - 1

        count = 0
        for i, word in enumerate(cand['parse']):

            if word.form.lower() not in config.stopwords:
                continue
            if i in align_dict.keys():
                continue

            wd_size = 3
            bwd = range(max(i - wd_size, 0), i)
            fwd = range(i + 1, min(i + 1 + wd_size, len(cand['parse'])))
            match_bwd = []
            match_fwd = []

            for j in sorted(bwd, reverse=True):
                if j in align_dict.keys():
                    if len(match_bwd) > 0 and align_dict[j] - match_bwd[-1] > 1:
                        break
                    match_bwd.append(align_dict[j])
                else:
                    break

            for k in fwd:
                if k in align_dict.keys():
                    if len(match_fwd) > 0 and align_dict[k] - match_fwd[-1] > 1:
                        break
                    match_fwd.append(align_dict[k])
                else:
                    break

            if len(match_bwd) > 1 and len(match_fwd) > 1:
                if self.is_sequence(match_bwd) and self.is_sequence(match_fwd):
                    count += 1

        AbstractFeature.set_value(self, count)

    @staticmethod
    def is_sequence(my_list):
        for i in range(len(my_list) - 1):
            if math.fabs(my_list[i] - my_list[i + 1]) > 1:
                return False
        return True


class MatchContextSimilarity(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'match_context_similarity')
        AbstractFeature.set_description(self, "Cosine similarity between words occurring in exactly matching contexts")

    def run(self, cand, ref):

        cleaner = CleanPunctuation()

        cl_cand, cl_ref = cleaner.clean_punctuation(cand, ref)

        align_dict = {}
        similarities = []

        for wpair in cl_cand['alignments'][0]:
            align_dict[wpair[0] - 1] = wpair[1] - 1

        for i, word in enumerate(cl_cand['parse']):

            if i in align_dict.keys():
                continue

            if word.form == 'the' or word.form == 'a':
                continue

            wd_size = 3
            bwd = range(max(i - wd_size, 0), i)
            fwd = range(i + 1, min(i + 1 + wd_size, len(cl_cand['parse'])))
            match_bwd = []
            match_fwd = []

            for j in sorted(bwd, reverse=True):
                if j in align_dict.keys():
                    if len(match_bwd) > 0 and align_dict[j] - match_bwd[-1] > 1:
                        break
                    match_bwd.append(align_dict[j])
                else:
                    break

            for k in fwd:
                if k in align_dict.keys():
                    if len(match_fwd) > 0 and align_dict[k] - match_fwd[-1] > 1:
                        break
                    match_fwd.append(align_dict[k])
                else:
                    break

            if len(match_bwd) >= 1 and len(match_fwd) >= 1:
                if self.is_sequence(match_bwd) and self.is_sequence(match_fwd):
                    if max(match_fwd) > max(match_bwd):
                        if match_bwd[0] + 1 in align_dict.values():
                            similarities.append(1.0)
                        else:
                            similarities.append(1 - dot(matutils.unitvec(cl_cand['word_vectors'][i]), matutils.unitvec(cl_ref['word_vectors'][match_bwd[0] + 1])))

        if len(similarities) == 0:
            AbstractFeature.set_value(self, 0.0)
        else:
            AbstractFeature.set_value(self, numpy.mean(similarities))

    @staticmethod
    def is_sequence(my_list):
        for i in range(len(my_list) - 1):
            if math.fabs(my_list[i] - my_list[i + 1]) > 1:
                return False
        return True


class FragPenalty(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'frag_penalty')
        AbstractFeature.set_description(self, "Fragmentation penalty from Meteor")

    def run(self, cand, ref):

        beta = 1.30 # parameters from Meteor rank task for Spanish
        gamma = 0.50

        chunckNumber = self.calculateChuncks(cand['alignments'][0])
        fragPenalty = 0.0

        if chunckNumber > 1:
            fragPenalty = gamma * pow(float(chunckNumber) / len(cand['alignments'][0]), beta)

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


class AvgWordQuest(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_word_quest')
        AbstractFeature.set_description(self, "Average on back-propagation behaviour for non-aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
             AbstractFeature.set_value(self, numpy.mean(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)

class MinWordQuest(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_word_quest')
        AbstractFeature.set_description(self, "Minimum on back-propagation behaviour for non-aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
             AbstractFeature.set_value(self, min(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class MaxWordQuest(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_word_quest')
        AbstractFeature.set_description(self, "Maximum on back-propagation behaviour for non-aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
             AbstractFeature.set_value(self, max(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class AvgWordQuestAlign(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_word_quest_align')
        AbstractFeature.set_description(self, "Average on back-propagation behaviour for aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
             AbstractFeature.set_value(self, numpy.mean(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)

class MinWordQuestAlign(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_word_quest_align')
        AbstractFeature.set_description(self, "Minimum on back-propagation behaviour for aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
             AbstractFeature.set_value(self, min(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class MaxWordQuestAlign(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_word_quest_align')
        AbstractFeature.set_description(self, "Maximum on back-off behaviour for aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
             AbstractFeature.set_value(self, max(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class AvgWordQuestBack(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_word_quest_back')
        AbstractFeature.set_description(self, "Average on back-off behaviour of backward lm for non-aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
             AbstractFeature.set_value(self, numpy.mean(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)

class MinWordQuestBack(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_word_quest_back')
        AbstractFeature.set_description(self, "Minimum on back-off behaviour of backward lm for non-aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
             AbstractFeature.set_value(self, min(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class MaxWordQuestBack(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_word_quest_back')
        AbstractFeature.set_description(self, "Maximum on back-propagation of backward lm behaviour for non-aligned words")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
             AbstractFeature.set_value(self, max(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class AvgPosProb(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pos_prob')
        AbstractFeature.set_description(self, "Average probability of pos language model")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['pos_lang_model'][i])

        if cnt > 0:
             AbstractFeature.set_value(self, numpy.mean(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class MinPosProb(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_pos_prob')
        AbstractFeature.set_description(self, "Min probability of pos language model")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['pos_lang_model'][i])

        if cnt > 0:
             AbstractFeature.set_value(self, min(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class MaxPosProb(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_pos_prob')
        AbstractFeature.set_description(self, "Max probability of pos language model")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['pos_lang_model'][i])

        if cnt > 0:
             AbstractFeature.set_value(self, max(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class CountPosProb(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_pos_prob')
        AbstractFeature.set_description(self, "Proportion of pos tags with low probability of pos language model")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['pos_lang_model'][i])

        threshold = 0.01
        if cnt > 0:
             AbstractFeature.set_value(self, len([x for x in back_props if x < threshold]))
        else:
             AbstractFeature.set_value(self, 0.0)


class PropPosProb(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pos_prob')
        AbstractFeature.set_description(self, "Proportion of pos tags with low probability of pos language model")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['pos_lang_model'][i])

        threshold = 0.01
        if cnt > 0:
             AbstractFeature.set_value(self, len([x for x in back_props if x < threshold])/float(len(cand['tokens'])))
        else:
             AbstractFeature.set_value(self, 0.0)


class AvgWordQuestLen(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_word_quest_len')
        AbstractFeature.set_description(self, "Average on the longest target n-gram")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
             AbstractFeature.set_value(self, numpy.mean(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)

class MinWordQuestLen(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_word_quest_len')
        AbstractFeature.set_description(self, "Minimum on the longest target n-gram")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
             AbstractFeature.set_value(self, min(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class MaxWordQuestLen(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_word_quest_len')
        AbstractFeature.set_description(self, "Maximum on the longest target n-gram")

    def run(self, cand, ref):

        back_props = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
             AbstractFeature.set_value(self, max(back_props))
        else:
             AbstractFeature.set_value(self, 0.0)


class LanguageModelProbabilityInformed(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'language_model_probability_informed')
        AbstractFeature.set_description(self, "In progress...")

    def run(self, cand, ref):

        word_probs = []

        for i, word in enumerate(cand['tokens']):

            prob, ngram = cand['language_model_word_features_informed'][i]

            if ngram == 0:
                continue

            if cand['tokens'][i] in config.punctuations or cand['tokens'][i].isdigit():
                continue

            penalty = 1.0
            penalty1 = 1.0

            if cand['tokens'][i] in config.stopwords:
                penalty1 = 0.3

            # if i + 1 not in [x[0] for x in cand['alignments'][0]]:
            #     penalty = 0.3

            word_probs.append(numpy.log10(prob * ngram * penalty * penalty1))

        AbstractFeature.set_value(self, numpy.sum(word_probs))
        # AbstractFeature.set_value(self, 10 ** (-(numpy.sum(word_probs)/count)))


class LangModProbSrilm(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_prob_srilm')
        AbstractFeature.set_description(self, "Language model log-probability using srilm")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][1])


class LangModPerlexSrilm(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_perplex_srilm')
        AbstractFeature.set_description(self, "Language model perplexity using srilm")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][2])


class LangModProb(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_prob')
        AbstractFeature.set_description(self, "Language model log-probability")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['quest_sentence']['1012'])


class LangModPerlex(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_perplex')
        AbstractFeature.set_description(self, "Language model perplexity")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['quest_sentence']['1013'])


class LangModPerlex2(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_perplex2')
        AbstractFeature.set_description(self, "Language model perplexity with no end sentence marker")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['quest_sentence']['1014'])


class CountFluencyErrors0(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_fluency_errors_0')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior < 5 but > 1")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 2 or cand['quest_word'][i]['WCE1041'] == 2:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        AbstractFeature.set_value(self, errors)


class CountFluencyErrors0Avg(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_fluency_errors_0_avg')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior < 5 but > 1")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                avg = (cand['quest_word'][i]['WCE1015'] + cand['quest_word'][i]['WCE1041']) / 2

                if 1 < avg < 3:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        AbstractFeature.set_value(self, errors)


class PropFluencyErrors0(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_fluency_errors_0')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior < 5 but > 1")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 2 or cand['quest_word'][i]['WCE1041'] == 2:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        result = errors/float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class PropFluencyErrors0Avg(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_fluency_errors_0_avg')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior < 5 but > 1")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                avg = (cand['quest_word'][i]['WCE1015'] + cand['quest_word'][i]['WCE1041']) / 2

                if 1 < avg < 3:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        result = errors/float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class CountFluencyErrors1(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_fluency_errors_1')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior >= 5 but < 7")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 3 or cand['quest_word'][i]['WCE1041'] == 3:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        AbstractFeature.set_value(self, errors)


class CountFluencyErrors1Avg(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_fluency_errors_1_avg')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior >= 5 but < 7")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                avg = (cand['quest_word'][i]['WCE1015'] + cand['quest_word'][i]['WCE1041']) / 2

                if 2 < avg < 4:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        AbstractFeature.set_value(self, errors)


class PropFluencyErrors1(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_fluency_errors_1')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior >= 5 but < 7")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 3 or cand['quest_word'][i]['WCE1041'] == 3:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        result = errors/float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class PropFluencyErrors1Avg(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_fluency_errors_1_avg')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior >= 5 but < 7")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                avg = (cand['quest_word'][i]['WCE1015'] + cand['quest_word'][i]['WCE1041']) / 2

                if 2 < avg < 4:

                    if i + 2 in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i + 1]['WCE1015'] == 1:
                        continue

                    if i in [x[0] for x in cand['alignments'][0]] and cand['quest_word'][i - 1]['WCE1015'] == 1:
                        continue

                    errors += 1

        result = errors/float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class CountNonAlignedOOV(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_oov')
        AbstractFeature.set_description(self, "Count of non-aligned out-of-vocabulary words (lm back-prop = 1)")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if cand['quest_word'][i]['WCE1015'] == 1:
                    oov += 1

        AbstractFeature.set_value(self, oov)


class PropNonAlignedOOV(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_oov')
        AbstractFeature.set_description(self, "Prop of non-aligned out-of-vocabulary words (lm back-prop = 1)")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if cand['quest_word'][i]['WCE1015'] == 1:
                    oov += 1

        if len(cand['alignments'][0]) != len(cand['tokens']):
            AbstractFeature.set_value(self, oov/float(len(cand['tokens']) - len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, 0)


class CountOOVSrilm(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_oov_srilm')
        AbstractFeature.set_description(self, "Count of out-of-vocabulary words using srilm")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][0])


class CountAllOOV(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_all_oov')
        AbstractFeature.set_description(self, "Count of out-of-vocabulary words (lm back-prop = 1)")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if cand['quest_word'][i]['WCE1015'] == 1:
                oov += 1

        AbstractFeature.set_value(self, oov)


class PropAllOOV(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_all_oov')
        AbstractFeature.set_description(self, "Proportion of out-of-vocabulary words (lm back-prop = 1)")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):
            if word.lower() in config.punctuations:
                continue

            if cand['quest_word'][i]['WCE1015'] == 1:
                oov += 1

        result = oov/float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class Bleu(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu')
        AbstractFeature.set_description(self, "Bleu score")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['bleu'])


class Meteor(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor')
        AbstractFeature.set_description(self, "Meteor score")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['meteor'])


class Cobalt(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'cobalt')
        AbstractFeature.set_description(self, "Cobalt score")

    def run(self, cand, ref):

        my_scorer = Scorer()
        word_level_scores = my_scorer.word_scores(cand['parse'], ref['parse'], cand['alignments'])
        score = my_scorer.sentence_score_cobalt(cand['parse'], ref['parse'], cand['alignments'], word_level_scores)
        AbstractFeature.set_value(self, score)


class VizWordQuest(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'viz_word_quest')
        AbstractFeature.set_description(self, "Visualize back-propagation behaviour")

    def run(self, cand, ref):

        print ' '.join([x.form for x in ref['parse']])
        print ' '.join([x.form for x in cand['parse']])

        if len(cand['tokens']) != len(cand['quest_word']):
            print "Sentence lengths do not match!"
            return

        for i, word in enumerate(cand['tokens']):
            print word + '\t' + str(cand['quest_word'][i]['WCE1015']) + '\t' + str(cand['quest_word'][i]['WCE1041']) +\
            '\t' + str((cand['quest_word'][i]['WCE1015'] + cand['quest_word'][i]['WCE1041'])/float(2))

        AbstractFeature.set_value(self, 'NaN')


class AvgDistanceNonAlignedTest(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_distance_non_aligned_test')
        AbstractFeature.set_description(self, "Avg distance between non-aligned words in candidate translation")

    def run(self, cand, ref):

        non_aligned = []
        distances = []

        for i, word in enumerate(cand['tokens'], start=1):

            if i not in [x[0] for x in cand['alignments'][0]]:

                non_aligned.append(i)

        if not len(non_aligned) > 1:
            AbstractFeature.set_value(self, 0)
            return

        for i, word in enumerate(sorted(non_aligned)):

            if i < len(non_aligned) - 1:
                distances.append(sorted(non_aligned)[i + 1] - word)
            else:
                break

        AbstractFeature.set_value(self, numpy.sum(distances)/len(distances))


class AvgDistanceNonAlignedRef(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_distance_non_aligned_ref')
        AbstractFeature.set_description(self, "Avg distance between non-aligned words in candidate translation")

    def run(self, cand, ref):

        non_aligned = []
        distances = []

        for i, word in enumerate(ref['tokens'], start=1):

            if i not in [x[0] for x in cand['alignments'][0]]:

                non_aligned.append(i)

        if not len(non_aligned) > 1:
            AbstractFeature.set_value(self, 0)
            return

        for i, word in enumerate(sorted(non_aligned)):

            if i < len(non_aligned) - 1:
                distances.append(sorted(non_aligned)[i + 1] - word)
            else:
                break

        AbstractFeature.set_value(self, numpy.sum(distances)/len(distances))


class MedianCosineDifference(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_difference')
        AbstractFeature.set_description(self, "Median cosine similarity between adjacent words in the target vs in the reference")

    def run(self, cand, ref):

        tgt_content_indexes = self.content_words_indexes(cand['tokens'])
        ref_content_indexes = self.content_words_indexes(ref['tokens'])

        if len(tgt_content_indexes) < 2 or len(ref_content_indexes) < 2:
            AbstractFeature.set_value(self, -1.0)
        else:
            tgt_similarities = self.cosine_similarities_sequence(cand, tgt_content_indexes)
            ref_similarities = self.cosine_similarities_sequence(ref, ref_content_indexes)
            AbstractFeature.set_value(self, numpy.median(tgt_similarities) - numpy.median(ref_similarities))

    @staticmethod
    def content_words_indexes(tokens):

        content_words_indexes = []

        for idx, token in enumerate(tokens):
            if token.lower() not in config.punctuations and token.lower() not in config.stopwords:
                content_words_indexes.append(idx)

        return content_words_indexes

    @staticmethod
    def cosine_similarities_sequence(sentence_object, content_words_indexes):

        similarities = []
        for i, idx in enumerate(content_words_indexes):

            if i >= len(content_words_indexes) - 1:
                break

            sim = dot(matutils.unitvec(sentence_object['word_vectors'][idx]), matutils.unitvec(sentence_object['word_vectors'][content_words_indexes[i + 1]]))
            similarities.append(sim)

        return similarities


class MedianCosineTarget(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_target')
        AbstractFeature.set_description(self, "Median cosine similarity between adjacent words in the target")

    def run(self, cand, ref):

        # print ' '.join(ref['tokens'])
        # print ' '.join(cand['tokens'])

        tgt_content_indexes = MedianCosineDifference.content_words_indexes(cand['tokens'])

        if len(tgt_content_indexes) < 2:
            AbstractFeature.set_value(self, -1.0)
        else:
            tgt_similarities = MedianCosineDifference.cosine_similarities_sequence(cand, tgt_content_indexes)
            AbstractFeature.set_value(self, numpy.median(tgt_similarities))


class MedianCosineReference(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_reference')
        AbstractFeature.set_description(self, "Median cosine similarity between adjacent words in the reference translation")

    def run(self, cand, ref):

        # print ' '.join(ref['tokens'])
        # print ' '.join(cand['tokens'])

        ref_content_indexes = MedianCosineDifference.content_words_indexes(ref['tokens'])

        if len(ref_content_indexes) < 2:
            AbstractFeature.set_value(self, -1.0)
        else:
            ref_similarities = MedianCosineDifference.cosine_similarities_sequence(ref, ref_content_indexes)
            AbstractFeature.set_value(self, numpy.median(ref_similarities))


class MedianCosineTargetNonAligned(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_target_non_aligned')
        AbstractFeature.set_description(self, "Median cosine similarity between adjacent words in the target for non-aligned words")

    def run(self, cand, ref):

        print ' '.join(ref['tokens'])
        print ' '.join(cand['tokens'])

        similarities = []

        content_indexes = MedianCosineDifference.content_words_indexes(cand['tokens'])

        if len(content_indexes) < 2:
            AbstractFeature.set_value(self, -1.0)
            return

        non_aligned_content_indexes = self.non_aligned_content_words_indexes(cand, content_indexes)

        if len(non_aligned_content_indexes) == 0:
            AbstractFeature.set_value(self, 1.0)
            return

        print "Non-aligned: " + ' '.join([cand['tokens'][x] for x in non_aligned_content_indexes])

        for index in non_aligned_content_indexes:
            context_words = self.get_context_words(index, content_indexes)

            if len(context_words) == 0:
                continue

            avg_similarity = self.cosine_similarity_neighbors(cand, index, context_words)

            similarities.append(avg_similarity)

        AbstractFeature.set_value(self, numpy.median(similarities))


    @staticmethod
    def get_context_words(target_word_idx, content_indexes):

        context_word_indexes = []
        for i, word_idx in enumerate(content_indexes):
            if word_idx == target_word_idx:
                if i > 0:
                    context_word_indexes.append(content_indexes[i - 1])
                if i < len(content_indexes) - 1:
                    context_word_indexes.append(content_indexes[i + 1])

        return context_word_indexes

    @staticmethod
    def cosine_similarity_neighbors(sentence_object, target_word_idx, context_words_idxs):

        similarities = []
        for context_words_idx in context_words_idxs:
            similarity = dot(matutils.unitvec(sentence_object['word_vectors'][target_word_idx]), matutils.unitvec(sentence_object['word_vectors'][context_words_idx]))
            similarities.append(similarity)
            print "Target word: " + sentence_object['tokens'][target_word_idx] + " similarity to context word " + sentence_object['tokens'][context_words_idx] + " is " + str(similarity)

        average = numpy.mean(similarities)
        print "Average context similarity for the word " + sentence_object['tokens'][target_word_idx] + " is " + str(average)
        return average

    @staticmethod
    def non_aligned_content_words_indexes(cand, content_indexes):

        non_aligned = []
        for index in content_indexes:
            if index + 1 not in [x[0] for x in cand['alignments'][0]]:
                non_aligned.append(index)

        return non_aligned


class CosineSimilarity(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'cosine_similarity')
        AbstractFeature.set_description(self, "Cosine similarity between candidate and reference sentence vectors")

    def run(self, cand, ref):

        AbstractFeature.set_value(self, dot(matutils.unitvec(cand['sent_vector']), matutils.unitvec(ref['sent_vector'])))


class EuclidDistance(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'euclid_distance')
        AbstractFeature.set_description(self, "Euclidean distance (L2) between candidate and reference sentence vectors")

    def run(self, cand, ref):

        AbstractFeature.set_value(self, distance.euclidean(cand['sent_vector'], ref['sent_vector']))


class L1Distance(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'l1_distance')
        AbstractFeature.set_description(self, "L1 distance between candidate and reference sentence vectors")

    def run(self, cand, ref):

        distances = []

        for i, val in enumerate(cand['sent_vector']):
            distances.append(math.fabs(val - ref['sent_vector'][i]))

        AbstractFeature.set_value(self, numpy.sum(distances))


class RandomNumber(AbstractFeature):

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'random_number')
        AbstractFeature.set_description(self, "Generates random numbers, may be used to test feature selection algorithms")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, numpy.random.uniform(0.0, 1.0))

