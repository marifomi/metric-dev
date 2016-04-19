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
from src.sent_bleu.sent_bleu import SentBleu
from collections import Counter

__author__ = 'marina'


###########################################<Common Alignment Features>##################################################
########################################################################################################################
class CountWordsCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_words_candidate')
        AbstractFeature.set_description(self, "Number of words in the candidate")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['tokens']))


class CountWordsReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_words_reference')
        AbstractFeature.set_description(self, "Number of words in the reference")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(ref['tokens']))


class CountContentCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_content_candidate')
        AbstractFeature.set_description(self, "Number of content words in the candidate")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        count = 0

        for word in cand['tokens']:
            if not word_sim.function_word(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountContentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_content_reference')
        AbstractFeature.set_description(self, "Number of content words in the reference")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        count = 0

        for word in ref['tokens']:
            if not word_sim.function_word(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountFunctionCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_function_candidate')
        AbstractFeature.set_description(self, "Number of function words in the candidate")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        count = 0

        for word in cand['tokens']:
            if word_sim.function_word(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountFunctionReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_function_reference')
        AbstractFeature.set_description(self, "Number of function words in the reference")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        count = 0

        for word in ref['tokens']:
            if word_sim.function_word(word):
                count += 1

        AbstractFeature.set_value(self, count)


class CountAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned')
        AbstractFeature.set_description(self, "Number of aligned words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['alignments'][0]))


class CountNonAlignedCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_candidate')
        AbstractFeature.set_description(self, "Number of non-aligned words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['tokens']) - len(cand['alignments'][0]))


class CountNonAlignedReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_reference')
        AbstractFeature.set_description(self, "Number of non-aligned words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(ref['tokens']) - len(ref['alignments'][0]))


class PropNonAlignedCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_candidate')
        AbstractFeature.set_description(self, "Proportion of non-aligned words in the candidate")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['tokens']) > 0:
            AbstractFeature.set_value(self,
                                      (len(cand['tokens']) - len(cand['alignments'][0])) / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class PropNonAlignedReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_reference')
        AbstractFeature.set_description(self, "Proportion of non-aligned words in the reference")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(ref['tokens']) > 0:
            AbstractFeature.set_value(self,
                                      (len(ref['tokens']) - len(cand['alignments'][0])) / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned words in the candidate")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['tokens']) > 0:
            AbstractFeature.set_value(self, len(cand['alignments'][0]) / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class PropAlignedReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words in the reference")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(ref['tokens']) > 0:
            AbstractFeature.set_value(self, len(cand['alignments'][0]) / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, 0)


class CountAlignedContent(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_content')
        AbstractFeature.set_description(self, "Count of aligned content words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for word in cand['alignments'][1]:
                if not word_sim.function_word(word[0]):
                    count += 1

            AbstractFeature.set_value(self, count)

        else:
            AbstractFeature.set_value(self, -1)


class CountAlignedFunction(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_aligned_function')
        AbstractFeature.set_description(self, "Count of aligned function words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for word in cand['alignments'][1]:
                if word_sim.function_word(word[0]):
                    count += 1

            AbstractFeature.set_value(self, count)

        else:
            AbstractFeature.set_value(self, -1)


class CountNonAlignedContentCandidate(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_content_candidate')
        AbstractFeature.set_description(self, "Count of non aligned content words in the candidate translation")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if not word_sim.function_word(word):
                    count += 1

            AbstractFeature.set_value(self, count)


class CountNonAlignedFunctionCandidate(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_function_candidate')
        AbstractFeature.set_description(self, "Count of non-aligned function words in the candidate translation")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if word_sim.function_word(word):
                    count += 1

            AbstractFeature.set_value(self, count)


class PropNonAlignedContent(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_content_candidate')
        AbstractFeature.set_description(self, "Proportion of non aligned content words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if not word_sim.function_word(word):
                    count += 1

        AbstractFeature.set_value(self, count / float(len(cand['tokens']) - len(cand['alignments'][0])))


class PropNonAlignedFunction(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_function_candidate')
        AbstractFeature.set_description(self, "Prop of non-aligned function words")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        count = 0
        for i, word in enumerate(cand['tokens']):
            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if word.lower() in config.cobalt_stopwords and word.lower() not in config.punctuations:
                    count += 1

        AbstractFeature.set_value(self, count / float(len(cand['tokens']) - len(cand['alignments'][0])))


class PropAlignedContentCandidate(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_content_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned content words in the candidate translation")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        aligned_words = [x[0] - 1 for x in cand['alignments'][0]]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(aligned_words)) / float(len(content_words)))


class PropAlignedFunctionCandidate(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_function_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned function words in the candidate translation")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(cand['tokens']) if word_sim.function_word(x)]
        aligned_words = [x[0] - 1 for x in cand['alignments'][0]]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(aligned_words)) / float(len(function_words)))


class PropAlignedContentReference(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_content_reference')
        AbstractFeature.set_description(self, "Proportion of aligned content words in the reference translation")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        aligned_words = [x[0] - 1 for x in ref['alignments'][0]]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(aligned_words)) / float(len(content_words)))


class PropAlignedFunctionReference(AbstractFeature):
    # Supposing content words can only be aligned to content words

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_aligned_function_reference')
        AbstractFeature.set_description(self, "Proportion of aligned function words in the reference translation")
        AbstractFeature.set_group(self, "alignment")

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(ref['tokens']) if word_sim.function_word(x)]
        aligned_words = [x[0] - 1 for x in ref['alignments'][0]]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(aligned_words)) / float(len(function_words)))


###########################################</Common Alignment Features>#################################################
########################################################################################################################

###########################################<Cobalt Lexical Similarity>##################################################
########################################################################################################################


class CobaltPropExactLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_exact_lex_exact_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with exact lexical match and exact POS match")
        AbstractFeature.set_group(self, "cobalt_lexical_similarity")

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountExactLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_exact_lex_exact_pos')
        AbstractFeature.set_description(self, "Count of aligned words with exact lexical match and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropSynLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_syn_lex_exact_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with synonym lexical match and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountSynLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_syn_lex_exact_pos')
        AbstractFeature.set_description(self, "Count of aligned words with synonym lexical match and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropParaLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_para_lex_exact_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with paraphrase lexical match and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountParaLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_para_lex_exact_pos')
        AbstractFeature.set_description(self,
                                        "Count of aligned words with paraphrase lexical match and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropExactLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_exact_lex_coarse_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with exact lexical match and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountExactLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_exact_lex_coarse_pos')
        AbstractFeature.set_description(self, "Count of aligned words with exact lexical match and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropSynLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_syn_lex_coarse_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with synonym lexical match and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' \
                        and word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountSynLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_syn_lex_coarse_pos')
        AbstractFeature.set_description(self, "Count of aligned words with synonym lexical match and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' \
                        and word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropParaLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_para_lex_coarse_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with paraphrase lexical match and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountParaLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_para_lex_coarse_pos')
        AbstractFeature.set_description(self,
                                        "Count of aligned words with paraphrase lexical match and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase':
                    if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse':
                        count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropSynLexDiffPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_syn_lex_diff_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with synonym lexical match and different POS")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountSynLexDiffPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_syn_lex_diff_pos')
        AbstractFeature.set_description(self, "Count of aligned words with synonym lexical match and different POS")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropParaLexDiffPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_para_lex_diff_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with paraphrase lexical match and different POS")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountParaLexDiffPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_para_lex_diff_pos')
        AbstractFeature.set_description(self, "Count of aligned words with paraphrase lexical match and different POS")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropDistribLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_distrib_lex_exact_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with distributional similarity and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountDistribLexExactPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_distrib_lex_exact_pos')
        AbstractFeature.set_description(self,
                                        "Count of aligned words with distributional similarity and exact POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional' \
                        and word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropDistribLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_distrib_lex_coarse_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with distributional similarity and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' \
                        and word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountDistribLexCoarsePos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_distrib_lex_coarse_pos')
        AbstractFeature.set_description(self,
                                        "Count of aligned words with distributional similarity and coarse POS match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Coarse' \
                        and word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropDistribLexDiffPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_distrib_lex_diff_pos')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned words with distributional similarity and different POS")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None' \
                        and word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountDistribLexDiffPos(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_distrib_lex_diff_pos')
        AbstractFeature.set_description(self, "Count of aligned words with distributional similarity and different POS")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]):
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None' \
                        and word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count)
        else:
            AbstractFeature.set_value(self, -1)


class PropPosExact(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pos_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact pos match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

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
            AbstractFeature.set_value(self, -1)


class PropPosCoarse(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pos_coarse')
        AbstractFeature.set_description(self, "Proportion of aligned words with coarse pos match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

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
            AbstractFeature.set_value(self, -1)


class PropPosDiff(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pos_diff')
        AbstractFeature.set_description(self, "Proportion of aligned words with different pos")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

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
            AbstractFeature.set_value(self, -1)


class PropLexExact(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_lex_exact')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lex match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class PropLexSyn(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_lex_syn')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lex match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Synonym':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class PropLexPara(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_lex_para')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase lex match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Paraphrase':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class PropLexDistrib(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_lex_distrib')
        AbstractFeature.set_description(self, "Proportion of aligned words with distrib lex match")
        AbstractFeature.set_group(self, 'cobalt_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for index in cand['alignments'][0]:
                word_candidate = cand['parse'][index[0] - 1]
                word_reference = ref['parse'][index[1] - 1]

                if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Distributional':
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


###########################################</Cobalt Lexical Similarity>##################################################
########################################################################################################################

###########################################<Cobalt Context Penalty>##################################################
########################################################################################################################

class AvgPenExactCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_exact_candidate')
        AbstractFeature.set_description(self, "Average CP for aligned words with exact match in the candidate"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

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


class AvgPenExactReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_exact_reference')
        AbstractFeature.set_description(self, "Average CP for aligned words with exact match in the reference"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')


    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

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


class AvgPenNonExactCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_non_exact_candidate')
        AbstractFeature.set_description(self, "Average CP for aligned words with non exact match in the candidate"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

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


class AvgPenNonExactReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_non_exact_reference')
        AbstractFeature.set_description(self, "Average CP for aligned words with non exact match in the reference"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

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


class PropPenExactCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_exact_candidate')
        AbstractFeature.set_description(self, "Proportion of words with CP with exact match in the candidate"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

                counter_words += 1

                if len(cand['alignments'][2][i]['srcDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
            AbstractFeature.set_value(self, counter_penalties / float(counter_words))
        else:
            AbstractFeature.set_value(self, 0.0)


class PropPenExactReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_exact_reference')
        AbstractFeature.set_description(self, "Proportion of words with CP with exact match in the reference"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'None':

                counter_words += 1

                if len(cand['alignments'][2][i]['tgtDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
            AbstractFeature.set_value(self, counter_penalties / float(counter_words))
        else:
            AbstractFeature.set_value(self, 0.0)


class PropPenNonExactCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_non_exact_candidate')
        AbstractFeature.set_description(self, "Proportion of words with CP with non exact match in the candidate"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

                counter_words += 1

                if len(cand['alignments'][2][i]['srcDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
            AbstractFeature.set_value(self, counter_penalties / float(counter_words))
        else:
            AbstractFeature.set_value(self, 0.0)


class PropPenNonExactReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_non_exact_reference')
        AbstractFeature.set_description(self, "Proportion of words with CP with non exact match in the reference"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        counter_words = 0
        counter_penalties = 0

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        for i, index in enumerate(cand['alignments'][0]):

            word_candidate = cand['parse'][index[0] - 1]
            word_reference = ref['parse'][index[1] - 1]

            if not word_sim.word_relatedness_feature(word_candidate, word_reference) == 'Exact' \
                    and not word_sim.comparePos(word_candidate.pos, word_reference.pos) == 'Exact':

                counter_words += 1

                if len(cand['alignments'][2][i]['tgtDiff']) > 0:
                    counter_penalties += 1

        if counter_words > 0:
            AbstractFeature.set_value(self, counter_penalties / float(counter_words))
        else:
            AbstractFeature.set_value(self, 0.0)


class AvgPenCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_candidate')
        AbstractFeature.set_description(self, "Average CP for the candidate translation"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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


class AvgPentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_pen_reference')
        AbstractFeature.set_description(self, "Average CP for aligned words in the reference translation"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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


class MinPenCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_pen_candidate')
        AbstractFeature.set_description(self, "Minimum CP for the candidate translation"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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


class MinPentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'min_pen_reference')
        AbstractFeature.set_description(self, "Minimum CP for aligned words in the reference translation"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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


class MaxPenCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_pen_candidate')
        AbstractFeature.set_description(self, "Max CP for the candidate translation"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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


class MaxPentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'max_pen_reference')
        AbstractFeature.set_description(self, "Max CP for aligned words in the reference translation"
                                              "(considered only for the words with CP > 0)")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter_penalties = 0.0
        counter_words = 0.0

        for i, index in enumerate(cand['alignments'][0]):

            counter_words += 1

            if len(cand['alignments'][2][i]['srcDiff']) > 0 or len(cand['alignments'][2][i]['tgtDiff']) > 0:
                counter_penalties += 1

        if counter_words > 0:
            AbstractFeature.set_value(self, counter_penalties / float(counter_words))
        else:
            AbstractFeature.set_value(self, 0.0)


class CountPen(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_pen')
        AbstractFeature.set_description(self, "Count of words with penalty over the number of aligned words")
        AbstractFeature.set_group(self, 'cobalt_context_penalty')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter_penalties = 0.0

        for i, index in enumerate(cand['alignments'][0]):

            if len(cand['alignments'][2][i]['srcDiff']) > 0 or len(cand['alignments'][2][i]['tgtDiff']) > 0:
                counter_penalties += 1

        AbstractFeature.set_value(self, counter_penalties)


class PropPenHigh(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_pen_high')
        AbstractFeature.set_description(self,
                                        "Prop of words with penalty higher than avg over the number of aligned words")

        # This should be higher than average context penalty estimated over some big dataset

    def run(self, cand, ref):

        my_scorer = Scorer()
        difference = 0.0
        context = 0.0
        penalties = []

        if len(cand['alignments'][0]) == 0:
            AbstractFeature.set_value(self, -1)
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


###########################################</Cobalt Context Penalty>##################################################
########################################################################################################################

#################################################<Meteor Decomposed>####################################################
########################################################################################################################


class FragmentationPenalty(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'fragmentation_penalty')
        AbstractFeature.set_description(self, "Fragmentation penalty from Meteor")
        AbstractFeature.set_group(self, 'meteor_fragmentation_penalty')

    def run(self, cand, ref):

        chunck_number = self.calculate_chuncks(cand['alignments'][0])
        frag_penalty = 0.0

        if chunck_number > 1:
            frag_penalty = float(chunck_number) / len(cand['alignments'][0])

        AbstractFeature.set_value(self, frag_penalty)

    def calculate_chuncks(self, alignments):

        sortedAlignments = sorted(alignments, key=lambda alignment: alignment[0])

        chunks = 0
        previousPair = None

        for pair in sortedAlignments:
            if previousPair == None or previousPair[0] != pair[0] - 1 or previousPair[1] != pair[1] - 1:
                chunks += 1
            previousPair = pair

        return chunks


class CountChunks(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_chunks')
        AbstractFeature.set_description(self, "Number of chunks")
        AbstractFeature.set_group(self, 'meteor_fragmentation_penalty')

    def run(self, cand, ref):

        chunck_number = self.calculate_chuncks(cand['alignments'][0])
        AbstractFeature.set_value(self, chunck_number)

    def calculate_chuncks(self, alignments):

        sortedAlignments = sorted(alignments, key=lambda alignment: alignment[0])

        chunks = 0
        previousPair = None

        for pair in sortedAlignments:
            if previousPair == None or previousPair[0] != pair[0] - 1 or previousPair[1] != pair[1] - 1:
                chunks += 1
            previousPair = pair

        return chunks


class MeteorPropExactCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_exact_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lexical match for Meteor"
                                              "for the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropExactReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_exact_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with exact lexical match for Meteor"
                                              "for the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropExactContentCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_exact_content_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned content words with exact lexical match for Meteor"
                                              "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        exact_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 0]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(exact_words)) / float(len(content_words)))


class MeteorPropExactContentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_exact_content_reference')
        AbstractFeature.set_description(self, "Proportion of aligned content words with exact lexical match for Meteor"
                                              "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        exact_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 0]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(exact_words)) / float(len(content_words)))


class MeteorPropExactFunctionCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_exact_function_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned function words with exact lexical match for Meteor"
                                              "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(cand['tokens']) if word_sim.function_word(x)]
        exact_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 0]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(exact_words)) / float(len(function_words)))


class MeteorPropExactFunctionReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_exact_function_reference')
        AbstractFeature.set_description(self, "Proportion of aligned function words with exact lexical match for Meteor"
                                              "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(ref['tokens']) if word_sim.function_word(x)]
        exact_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 0]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(exact_words)) / float(len(function_words)))


class MeteorPropFuzzyCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_fuzzy_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned words with non-exact lexical match for Meteor"
                                              "for the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] != 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropFuzzyReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_fuzzy_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with non-exact lexical match for Meteor"
                                              "for the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] != 0:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropFuzzyContentCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_fuzzy_content_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned content words with fuzzy lexical match for Meteor"
                                              "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        fuzzy_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] != 0]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(fuzzy_words)) / float(len(content_words)))


class MeteorPropFuzzyContentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_fuzzy_content_reference')
        AbstractFeature.set_description(self, "Proportion of aligned content words with fuzzy lexical match for Meteor"
                                              "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        fuzzy_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] != 0]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(fuzzy_words)) / float(len(content_words)))


class MeteorPropFuzzyFunctionCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_fuzzy_function_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned function words with fuzzy lexical match for Meteor"
                                              "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        fuzzy_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] != 0]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(fuzzy_words)) / float(len(function_words)))


class MeteorPropFuzzyFunctionReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_fuzzy_function_reference')
        AbstractFeature.set_description(self, "Proportion of aligned function words with fuzzy lexical match for Meteor"
                                              "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        fuzzy_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] != 0]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(fuzzy_words)) / float(len(function_words)))


class MeteorPropStemCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_stem_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned words with stem match for Meteor"
                                              "for the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 1:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropStemReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_stem_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with stem match for Meteor"
                                              "for the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 1:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropStemContentCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_stem_content_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned content words with stem lexical match for Meteor"
                                              "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        stem_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 1]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(stem_words)) / float(len(content_words)))


class MeteorPropStemContentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_stem_content_reference')
        AbstractFeature.set_description(self, "Proportion of aligned content words with stem lexical match for Meteor"
                                              "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        stem_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 1]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(stem_words)) / float(len(content_words)))


class MeteorPropStemFunctionCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_stem_function_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned function words with stem lexical match for Meteor"
                                              "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        stem_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 1]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(stem_words)) / float(len(function_words)))


class MeteorPropStemFunctionReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_stem_function_reference')
        AbstractFeature.set_description(self, "Proportion of aligned function words with stem lexical match for Meteor"
                                              "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        stem_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 1]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(stem_words)) / float(len(function_words)))


class MeteorPropSynonymCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_synonym_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lexical match for Meteor"
                                              "for the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 2:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropSynonymReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_synonym_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with synonym lexical match for Meteor"
                                              "for the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 2:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropSynonymContentCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_synonym_content_candidate')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned content words with synonym lexical match for Meteor"
                                        "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        synonym_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 2]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(synonym_words)) / float(len(content_words)))


class MeteorPropSynonymContentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_synonym_content_reference')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned content words with synonym lexical match for Meteor"
                                        "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        synonym_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 2]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(content_words).intersection(synonym_words)) / float(len(content_words)))


class MeteorPropSynonymFunctionCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_synonym_function_candidate')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned function words with synonym lexical match for Meteor"
                                        "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        synonym_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 2]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(synonym_words)) / float(len(function_words)))


class MeteorPropSynonymFunctionReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_synonym_function_reference')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned function words with synonym lexical match for Meteor"
                                        "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        synonym_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 2]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self,
                                      len(set(function_words).intersection(synonym_words)) / float(len(function_words)))


class MeteorPropParaphraseCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_paraphrase_candidate')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase match for Meteor"
                                              "for the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 3:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropParaphraseReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_paraphrase_reference')
        AbstractFeature.set_description(self, "Proportion of aligned words with paraphrase match for Meteor"
                                              "for the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        if len(cand['alignments'][0]) > 0:
            count = 0

            for i in range(len(cand['alignments'][0])):

                if cand['alignments'][2][i] == 3:
                    count += 1

            AbstractFeature.set_value(self, count / float(len(ref['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class MeteorPropParaphraseContentCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_paraphrase_content_candidate')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned content words with paraphrase lexical match for Meteor"
                                        "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        paraphrase_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 3]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, len(set(content_words).intersection(paraphrase_words)) / float(
                len(content_words)))


class MeteorPropParaphraseContentReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_paraphrase_content_reference')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned content words with paraphrase lexical match for Meteor"
                                        "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        content_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        paraphrase_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 3]

        if len(content_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, len(set(content_words).intersection(paraphrase_words)) / float(
                len(content_words)))


class MeteorPropParaphraseFunctionCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_paraphrase_function_candidate')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned function words with paraphrase lexical match for Meteor"
                                        "in the candidate translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(cand['tokens']) if not word_sim.function_word(x)]
        paraphrase_words = [x[0] - 1 for i, x in enumerate(cand['alignments'][0]) if cand['alignments'][2][i] == 3]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, len(set(function_words).intersection(paraphrase_words)) / float(
                len(function_words)))


class MeteorPropParaphraseFunctionReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'meteor_prop_paraphrase_function_reference')
        AbstractFeature.set_description(self,
                                        "Proportion of aligned function words with paraphrase lexical match for Meteor"
                                        "in the reference translation")
        AbstractFeature.set_group(self, 'meteor_lexical_similarity')

    def run(self, cand, ref):

        function_words = [i for i, x in enumerate(ref['tokens']) if not word_sim.function_word(x)]
        paraphrase_words = [x[0] - 1 for i, x in enumerate(ref['alignments'][0]) if ref['alignments'][2][i] == 3]

        if len(function_words) == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, len(set(function_words).intersection(paraphrase_words)) / float(
                len(function_words)))


#################################################</Meteor Decomposed>###################################################
########################################################################################################################

#################################################<BLEU Decomposed>######################################################
########################################################################################################################

class BleuPrecisionUnigram(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu_precision_unigram')
        AbstractFeature.set_description(self, "Bleu unigram modified precision")
        AbstractFeature.set_group(self, "bleu_lexical_similarity")

    def run(self, cand, ref):

        matches, total = SentBleu.modified_precision([c.lower() for c in cand['tokens']],
                                                     [[r.lower() for r in ref['tokens']]], 1)

        if total == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, matches / float(total))


class BleuPrecisionBigram(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu_precision_bigram')
        AbstractFeature.set_description(self, "Bleu bigram modified precision")
        AbstractFeature.set_group(self, "bleu_lexical_similarity")

    def run(self, cand, ref):

        matches, total = SentBleu.modified_precision([c.lower() for c in cand['tokens']],
                                                     [[r.lower() for r in ref['tokens']]], 2)

        if total == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, matches / float(total))


class BleuPrecisionTrigram(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu_precision_trigram')
        AbstractFeature.set_description(self, "Bleu trigram modified precision")
        AbstractFeature.set_group(self, "bleu_lexical_similarity")

    def run(self, cand, ref):

        matches, total = SentBleu.modified_precision([c.lower() for c in cand['tokens']],
                                                     [[r.lower() for r in ref['tokens']]], 3)

        if total == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, matches / float(total))


class BleuPrecisionFourgram(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu_precision_fourgram')
        AbstractFeature.set_description(self, "Bleu fourgram modified precision")
        AbstractFeature.set_group(self, "bleu_lexical_similarity")

    def run(self, cand, ref):

        matches, total = SentBleu.modified_precision([c.lower() for c in cand['tokens']],
                                                     [[r.lower() for r in ref['tokens']]], 4)

        if total == 0:
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, matches / float(total))


class BleuBrevityPenalty(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu_brevity_penalty')
        AbstractFeature.set_description(self, "Bleu brevity penalty")
        AbstractFeature.set_group(self, "bleu_brevity_penalty")

    def run(self, cand, ref):
        bp = SentBleu.brevity_penalty(cand['tokens'], [ref['tokens']])
        AbstractFeature.set_value(self, bp)


#################################################</BLEU Decomposed>######################################################
########################################################################################################################

#################################################<Fluency Features>######################################################
########################################################################################################################


class BackoffNonAlignedAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_non_aligned_avg')
        AbstractFeature.set_description(self, "Average on backoff behaviour for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.mean(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffNonAlignedMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_non_aligned_min')
        AbstractFeature.set_description(self, "Minimum on backoff behaviour for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, min(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffNonAlignedMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_non_aligned_max')
        AbstractFeature.set_description(self, "Maximum on backoff behaviour for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffNonAlignedMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_non_aligned_median')
        AbstractFeature.set_description(self, "Median on backoff behaviour for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffNonAlignedMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_non_aligned_mode')
        AbstractFeature.set_description(self, "Mode on backoff behaviour for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(backoffs)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class BackoffBackNonAlignedAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_non_aligned_avg')
        AbstractFeature.set_description(self, "Average on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.mean(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackNonAlignedMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_non_aligned_min')
        AbstractFeature.set_description(self, "Minimum on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.min(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackNonAlignedMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_non_aligned_max')
        AbstractFeature.set_description(self, "Maximum on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackNonAlignedMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_non_aligned_median')
        AbstractFeature.set_description(self, "Median on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.median(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackNonAlignedMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_non_aligned_mode')
        AbstractFeature.set_description(self, "Mode on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(backoffs)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class BackoffAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_avg')
        AbstractFeature.set_description(self, "Average on backoff behaviour for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []

        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.mean(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_min')
        AbstractFeature.set_description(self, "Minimum on backoff behaviour for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, min(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_max')
        AbstractFeature.set_description(self, "Maximum on backoff behaviour for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_median')
        AbstractFeature.set_description(self, "Median on backoff behaviour for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt > 0:
            AbstractFeature.set_value(self, max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_mode')
        AbstractFeature.set_description(self, "Mode on backoff behaviour for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1015'])

        if cnt == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(backoffs)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class BackoffBackAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_avg')
        AbstractFeature.set_description(self, "Average on back-off behaviour of backward lm for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.mean(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_min')
        AbstractFeature.set_description(self, "Minimum on back-off behaviour of backward lm for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.min(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_max')
        AbstractFeature.set_description(self, "Maximum on back-off behaviour of backward lm for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_median')
        AbstractFeature.set_description(self, "Median on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.median(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class BackoffBackMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_back_mode')
        AbstractFeature.set_description(self, "Mode on back-off behaviour of backward lm for non-aligned words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1041'])

        if cnt == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(backoffs)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class LongestNgramNonAlignedAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_non_aligned_avg')
        AbstractFeature.set_description(self, "Average on the longest candidate n-gram for non-alinged words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.mean(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramNonAlignedMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_non_aligned_min')
        AbstractFeature.set_description(self, "Minimum on the longest candidate n-gram for non-alinged words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.min(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramNonAlignedMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_non_aligned_max')
        AbstractFeature.set_description(self, "Maximum on the longest candidate n-gram for non-alinged words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramNonAlignedMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_non_aligned_median')
        AbstractFeature.set_description(self, "Median on the longest candidate n-gram for non-alinged words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.median(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramNonAlignedMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_non_aligned_mode')
        AbstractFeature.set_description(self, "Mode on the longest candidate n-gram for non-alinged words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(backoffs)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class LongestNgramAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_avg')
        AbstractFeature.set_description(self, "Average on the longest candidate n-gram for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.mean(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_min')
        AbstractFeature.set_description(self, "Minimum on the longest candidate n-gram for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if cand['tokens'][i] in config.stopwords:
                penalty1 = 0.3

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.min(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_max')
        AbstractFeature.set_description(self, "Maximum on the longest candidate n-gram for words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.max(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_median')
        AbstractFeature.set_description(self, "Median on the longest candidate n-gram for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt > 0:
            AbstractFeature.set_value(self, numpy.median(backoffs))
        else:
            AbstractFeature.set_value(self, -1)


class LongestNgramMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'longest_ngram_mode')
        AbstractFeature.set_description(self, "Mode on the longest candidate n-gram for all words")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        backoffs = []
        cnt = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            cnt += 1
            backoffs.append(cand['quest_word'][i]['WCE1037'])

        if cnt == 0:
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(backoffs)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class BackoffDirectAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_direct_avg')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmean(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmean(ngram_lengths))


class BackoffDirectMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_direct_median')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmedian(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmedian(ngram_lengths))


class BackoffDirectMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_direct_min')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmin(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmin(ngram_lengths))


class BackoffDirectMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_direct_max')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmax(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmax(ngram_lengths))


class BackoffDirectMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'backoff_direct_mode')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        counter = Counter(ngram_lengths)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class BackDirectNonAlignedAvg(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'back_direct_non_aligned_avg')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmean(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmean(ngram_lengths))


class BackDirectNonAlignedMedian(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'back_direct_non_aligned_median')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmedian(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmedian(ngram_lengths))


class BackDirectNonAlignedMin(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'back_direct_non_aligned_min')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmin(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmin(ngram_lengths))


class BackDirectNonAlignedMax(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'back_direct_non_aligned_max')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmax(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        AbstractFeature.set_value(self, numpy.nanmax(ngram_lengths))


class BackDirectNonAlignedMode(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'back_direct_non_aligned_mode')
        AbstractFeature.set_description(self, "")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        ngram_lengths = []

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            ngram_lengths.append(cand['language_model_word_features'][i][1])

        if len(ngram_lengths) == 0 or numpy.isnan(numpy.nanmedian(ngram_lengths)):
            AbstractFeature.set_value(self, -1)
            return

        counter = Counter(ngram_lengths)
        counter_sorted = sorted(counter.most_common(), key=lambda x: (x[1], x[0]), reverse=True)

        if counter_sorted[0][1] == 1:
            AbstractFeature.set_value(self, 0.0)
        elif numpy.isnan(counter_sorted[0][0]):
            AbstractFeature.set_value(self, -1)
        else:
            AbstractFeature.set_value(self, counter_sorted[0][0])


class CountShortNgramNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_short_ngram_non_aligned')
        AbstractFeature.set_description(self, "In progress...")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        count = 0

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            if cand['language_model_word_features'][i][1] == 1:
                count += 1

        AbstractFeature.set_value(self, count)


class PropShortNgramNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_short_ngram_non_aligned')
        AbstractFeature.set_description(self, "In progress...")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        count = 0

        if len(cand['alignments']) == len(cand['tokens']):
            AbstractFeature.set_value(self, -1)
            return

        for i, word in enumerate(cand['tokens']):

            if i + 1 in [x[0] for x in cand['alignments'][0]]:
                continue

            if cand['language_model_word_features'][i][1] == 1:
                count += 1

        AbstractFeature.set_value(self, count / float(len(cand['tokens']) - len(cand['alignments'])))


class PropShortNgram(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_short_ngram')
        AbstractFeature.set_description(self, "In progress...")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        count = 0

        for i, word in enumerate(cand['tokens']):

            if cand['language_model_word_features'][i][1] == 1:
                count += 1

        AbstractFeature.set_value(self, count / float(len(cand['tokens'])))


class CountShortNgram(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_short_ngram')
        AbstractFeature.set_description(self, "In progress...")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        count = 0

        for i, word in enumerate(cand['tokens']):

            if cand['language_model_word_features'][i][1] == 1:
                count += 1

        AbstractFeature.set_value(self, count)


class CountBackoffLowNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_backoff_low_non_aligned')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior < 5 but > 1")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 2 or cand['quest_word'][i]['WCE1041'] == 2:
                    errors += 1

        AbstractFeature.set_value(self, errors)


class PropBackoffLowNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_backoff_low_non_aligned')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior < 5 but > 1")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 2 or cand['quest_word'][i]['WCE1041'] == 2:
                    errors += 1

        result = errors / float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class CountBackoffMediumNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_backoff_medium_non_aligned')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior >= 5 but < 7")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 3 or cand['quest_word'][i]['WCE1041'] == 3:
                    errors += 1

        AbstractFeature.set_value(self, errors)


class PropBackoffMediumNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_backoff_medium_non_aligned')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior >= 5 but < 7")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 3 or cand['quest_word'][i]['WCE1041'] == 3:
                    errors += 1

        result = errors / float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class CountBackoffHighNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_backoff_high_non_aligned')
        AbstractFeature.set_description(self, "Count of non-aligned words with back-off behavior == 7")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 4 or cand['quest_word'][i]['WCE1041'] == 4:
                    errors += 1

        AbstractFeature.set_value(self, errors)


class PropBackoffHighNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_backoff_high_non_aligned')
        AbstractFeature.set_description(self, "Proportion of non-aligned words with back-off behavior == 7")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        errors = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:

                if cand['quest_word'][i]['WCE1015'] == 4 or cand['quest_word'][i]['WCE1041'] == 4:
                    errors += 1

        result = errors / float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


class LangModProbSrilm(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_prob_srilm')
        AbstractFeature.set_description(self, "Language model log-probability using srilm")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][1])


class LangModPerlexSrilm(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_perplex_srilm')
        AbstractFeature.set_description(self, "Language model perplexity using srilm")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][2])


class LangModProb(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_prob')
        AbstractFeature.set_description(self, "Language model log-probability")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['quest_sentence']['1012'])


class LangModPerlex(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_perplex')
        AbstractFeature.set_description(self, "Language model perplexity")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['quest_sentence']['1013'])


class LangModPerlex2(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lang_mod_perplex2')
        AbstractFeature.set_description(self, "Language model perplexity with no end sentence marker")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['quest_sentence']['1014'])


class CountNonAlignedOOV(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_non_aligned_oov')
        AbstractFeature.set_description(self, "Count of non-aligned out-of-vocabulary words (lm backoff = 1)")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if cand['quest_word'][i]['WCE1015'] == 1:
                    oov += 1

        AbstractFeature.set_value(self, oov)


class PropNonAlignedOOV(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_non_aligned_oov')
        AbstractFeature.set_description(self, "Prop of non-aligned out-of-vocabulary words (lm backoff = 1)")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                if cand['quest_word'][i]['WCE1015'] == 1:
                    oov += 1

        if len(cand['alignments'][0]) != len(cand['tokens']):
            AbstractFeature.set_value(self, oov / float(len(cand['tokens']) - len(cand['alignments'][0])))
        else:
            AbstractFeature.set_value(self, -1)


class CountOOVSrilm(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_oov_srilm')
        AbstractFeature.set_description(self, "Count of out-of-vocabulary words using srilm")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][0])


class PropOOVSrilm(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_oov_srilm')
        AbstractFeature.set_description(self, "Proportion of out-of-vocabulary words using srilm")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['language_model_sentence_features'][0] / float(len(cand['tokens'])))


class CountOOV(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_oov')
        AbstractFeature.set_description(self, "Count of out-of-vocabulary words (lm backoff = 1)")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        oov = 0
        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if cand['quest_word'][i]['WCE1015'] == 1:
                oov += 1

        AbstractFeature.set_value(self, oov)


class PropOOV(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'prop_oov')
        AbstractFeature.set_description(self, "Proportion of out-of-vocabulary words (lm back-prop = 1)")
        AbstractFeature.set_group(self, "fluency_features")

    def run(self, cand, ref):

        oov = 0

        for i, word in enumerate(cand['tokens']):

            if word.lower() in config.punctuations:
                continue

            if word.lower().isdigit():
                continue

            if cand['quest_word'][i]['WCE1015'] == 1:
                oov += 1

        result = oov / float(len(cand['tokens']))
        AbstractFeature.set_value(self, result)


#################################################</Fluency Features>######################################################
########################################################################################################################


class ContextMatch(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'context_match')
        AbstractFeature.set_description(self,
                                        "Number of cases with non-aligned function words in exactly matching contexts")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):

        align_dict = {}

        for wpair in cand['alignments'][0]:
            align_dict[wpair[0] - 1] = wpair[1] - 1

        count = 0
        for i, word in enumerate(cand['parse']):

            if not word_sim.function_word(word.form):
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
        AbstractFeature.set_group(self, "miscellaneous")

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
                            similarities.append(1 - dot(matutils.unitvec(cl_cand['word_vectors'][i]),
                                                        matutils.unitvec(cl_ref['word_vectors'][match_bwd[0] + 1])))

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


class LengthsRatio(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'lengths_ratio')
        AbstractFeature.set_description(self, "Ratio of candidate and reference sentence lengths")
        AbstractFeature.set_group(self, "miscellaneous")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, len(cand['tokens']) / float(len(ref['tokens'])))


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

            if word.lower().isdigit():
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

            if word.lower().isdigit():
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

            if word.lower().isdigit():
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

            if word.lower().isdigit():
                continue

            if i + 1 not in [x[0] for x in cand['alignments'][0]]:
                cnt += 1
                back_props.append(cand['pos_lang_model'][i])

        threshold = 0.01
        if cnt > 0:
            AbstractFeature.set_value(self, len([x for x in back_props if x < threshold]))
        else:
            AbstractFeature.set_value(self, -1)


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
            AbstractFeature.set_value(self, len([x for x in back_props if x < threshold]) / float(len(cand['tokens'])))
        else:
            AbstractFeature.set_value(self, -1)


class Bleu(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'bleu')
        AbstractFeature.set_description(self, "BLEU score")

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


class Cobalt(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'cobalt_from_file')
        AbstractFeature.set_description(self, "Cobalt scores read from file")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, cand['cobalt'])


class VizWordQuest(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'viz_word_quest')
        AbstractFeature.set_description(self, "Visualize back-propagation behaviour")

    def run(self, cand, ref):

        print(' '.join([x.form for x in ref['parse']]))
        print(' '.join([x.form for x in cand['parse']]))

        if len(cand['tokens']) != len(cand['quest_word']):
            print("Sentence lengths do not match!")
            return

        for i, word in enumerate(cand['tokens']):
            print(word + '\t' + str(cand['quest_word'][i]['WCE1015']) + '\t' + str(cand['quest_word'][i]['WCE1041']) + \
                  '\t' + str((cand['quest_word'][i]['WCE1015'] + cand['quest_word'][i]['WCE1041']) / float(2)))

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

        AbstractFeature.set_value(self, numpy.sum(distances) / len(distances))


class AvgDistanceNonAlignedReference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'avg_distance_non_aligned_reference')
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

        AbstractFeature.set_value(self, numpy.sum(distances) / len(distances))


class MedianCosineDifference(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_difference')
        AbstractFeature.set_description(self,
                                        "Median cosine similarity between adjacent words in the candidate vs in the reference")

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
            if token.lower() not in config.punctuations and token.lower() not in config.cobalt_stopwords:
                content_words_indexes.append(idx)

        return content_words_indexes

    @staticmethod
    def cosine_similarities_sequence(sentence_object, content_words_indexes):

        similarities = []
        for i, idx in enumerate(content_words_indexes):

            if i >= len(content_words_indexes) - 1:
                break

            sim = dot(matutils.unitvec(sentence_object['word_vectors'][idx]),
                      matutils.unitvec(sentence_object['word_vectors'][content_words_indexes[i + 1]]))
            similarities.append(sim)

        return similarities


class MedianCosineCandidate(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_candidate')
        AbstractFeature.set_description(self, "Median cosine similarity between adjacent words in the candidate")

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
        AbstractFeature.set_description(self,
                                        "Median cosine similarity between adjacent words in the reference translation")

    def run(self, cand, ref):

        # print ' '.join(ref['tokens'])
        # print ' '.join(cand['tokens'])

        ref_content_indexes = MedianCosineDifference.content_words_indexes(ref['tokens'])

        if len(ref_content_indexes) < 2:
            AbstractFeature.set_value(self, -1.0)
        else:
            ref_similarities = MedianCosineDifference.cosine_similarities_sequence(ref, ref_content_indexes)
            AbstractFeature.set_value(self, numpy.median(ref_similarities))


class MedianCosineCandidateNonAligned(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'median_cosine_candidate_non_aligned')
        AbstractFeature.set_description(self,
                                        "Median cosine similarity between adjacent words in the candidate for non-aligned words")

    def run(self, cand, ref):

        print(' '.join(ref['tokens']))
        print(' '.join(cand['tokens']))

        similarities = []

        content_indexes = MedianCosineDifference.content_words_indexes(cand['tokens'])

        if len(content_indexes) < 2:
            AbstractFeature.set_value(self, -1.0)
            return

        non_aligned_content_indexes = self.non_aligned_content_words_indexes(cand, content_indexes)

        if len(non_aligned_content_indexes) == 0:
            AbstractFeature.set_value(self, 1.0)
            return

        print("Non-aligned: " + ' '.join([cand['tokens'][x] for x in non_aligned_content_indexes]))

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
            similarity = dot(matutils.unitvec(sentence_object['word_vectors'][target_word_idx]),
                             matutils.unitvec(sentence_object['word_vectors'][context_words_idx]))
            similarities.append(similarity)
            print("Candidate word: " + sentence_object['tokens'][target_word_idx] + " similarity to context word " +
                  sentence_object['tokens'][context_words_idx] + " is " + str(similarity))

        average = numpy.mean(similarities)
        print("Average context similarity for the word " + sentence_object['tokens'][target_word_idx] + " is " + str(
            average))
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
        AbstractFeature.set_value(self,
                                  dot(matutils.unitvec(cand['sent_vector']), matutils.unitvec(ref['sent_vector'])))


class EuclidDistance(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'euclid_distance')
        AbstractFeature.set_description(self,
                                        "Euclidean distance (L2) between candidate and reference sentence vectors")

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
        AbstractFeature.set_description(self,
                                        "Generates random numbers, may be used to test feature selection algorithms")

    def run(self, cand, ref):
        AbstractFeature.set_value(self, numpy.random.uniform(0.0, 1.0))


class CountWordsAlignedInWrongOrderRef(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_words_aligned_in_wrong_order_ref')
        AbstractFeature.set_description(self, 'Returns the number of cases when words are '
                                              'aligned not in their order in the sentence on reference side')

    def run(self, cand, ref):
        count = 0

        prev = None
        for a in ref['alignments'][0]:
            if prev is not None and prev > a[1]:
                count += 1
            prev = a[1]

        AbstractFeature.set_value(self, count)


class CountWordsAlignedInWrongOrderCand(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'count_words_aligned_in_wrong_order_cand')
        AbstractFeature.set_description(self, 'Returns the number of cases when words are '
                                              'aligned not in their order in the sentence on candidate side')

    def run(self, cand, ref):
        count = 0

        prev = None
        for a in cand['alignments'][0]:
            if prev is not None and prev > a[0]:
                count += 1

            prev = a[0]

        AbstractFeature.set_value(self, count)