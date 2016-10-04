from alignment.aligner_config import AlignerConfig
from utils import load_resources
from alignment.util import *


__author__ = 'anton'


class AlignerStanford(object):

    config = None
    alignments = []
    source_indices_aligned = []
    target_indices_aligned = []

    def __init__(self, language):
        self.config = AlignerConfig(language)

        if "paraphrases" in self.config.selected_lexical_resources:
            load_resources.load_ppdb(self.config.path_to_ppdb)
        if "distributional" in self.config.selected_lexical_resources:
            load_resources.load_word_vectors(self.config.path_to_vectors)

    def _add_to_alignments(self, source_index, target_index):
        self.alignments.append([source_index, target_index])
        self.source_indices_aligned.append(source_index)
        self.target_indices_aligned.append(target_index)

    def _align_ending_punctuation(self, source, target):
        if (source[len(source) - 1].is_sentence_ending_punctuation() and target[len(target) - 1].is_sentence_ending_punctuation())\
                or source[len(source) - 1].form == source[len(source) - 1].form:
            self._add_to_alignments(len(source), len(target))
        elif source[len(source) - 2].is_sentence_ending_punctuation() and target[len(target) - 1].is_sentence_ending_punctuation():
            self._add_to_alignments(len(source) - 1, len(target))
        elif source[len(source) - 1].is_sentence_ending_punctuation() and target[len(target) - 2].is_sentence_ending_punctuation():
            self._add_to_alignments(len(source), len(target) - 1)
        elif source[len(source) - 2].is_sentence_ending_punctuation() and target[len(target) - 2].is_sentence_ending_punctuation():
            self._add_to_alignments(len(source) -1, len(target) - 1)

        return

    def _align_contignuous_sublists(self, source, target):
        sublists = find_all_common_contiguous_sublists(source, target, True)

        for item in sublists:
            only_stopwords = True
            for jtem in item:
                if jtem not in cobalt_stopwords and jtem not in punctuations:
                    only_stopwords = False
                    break
            if len(item[0]) >= 2 and not only_stopwords:
                for j in range(len(item[0])):
                    if item[0][j]+1 not in self.source_indices_aligned \
                            and item[1][j]+1 not in self.source_indices_aligned \
                            and [item[0][j]+1, item[1][j]+1] not in self.alignments:
                        self._add_to_alignments(item[0][j]+1, item[1][j]+1)

    def align(self, source, target):

        self.alignments = []
        self.source_indices_aligned = []
        self.target_indices_aligned = []

        self._align_ending_punctuation(source, target)

        self._align_contignuous_sublists(source, target)

        return self.alignments

