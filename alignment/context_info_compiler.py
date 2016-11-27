from alignment.aligner_stanford import AlignerStanford
from alignment.aligner_config import AlignerConfig
from utils import load_resources
from utils.word_sim import *

__author__ = 'anton'


class ContextInfoCompiler(AlignerStanford):

    def __init__(self, language):
        self.config = AlignerConfig(language)

        if "paraphrases" in self.config.selected_lexical_resources:
            load_resources.load_ppdb(self.config.path_to_ppdb)
        if "distributional" in self.config.selected_lexical_resources:
            load_resources.load_word_vectors(self.config.path_to_vectors)

    def _compare_aligned_nodes(self, source, target, pos, opposite, relation_direction, alignments):
        # search for nodes in common or equivalent function

        relative_alignments = []
        word_similarities = []

        for word1 in source:
            for word2 in target:

                if ((word1.index, word2.index) in alignments or (word1.index == 0 and word2.index == 0)) and (
                    (word1.dep == word2.dep) or
                        ((pos != '' and relation_direction != 'child_parent') and (
                            self._is_similar(word1.dep, word2.dep, pos, 'noun', opposite, relation_direction) or
                            self._is_similar(word1.dep, word2.dep, pos, 'verb', opposite, relation_direction) or
                            self._is_similar(word1.dep, word2.dep, pos, 'adjective', opposite, relation_direction) or
                            self._is_similar(word1.dep, word2.dep, pos, 'adverb', opposite, relation_direction))) or
                        ((pos != '' and relation_direction == 'child_parent') and (
                            self._is_similar(word1.dep, word2.dep, pos, 'noun', opposite, relation_direction) or
                            self._is_similar(word1.dep, word2.dep, pos, 'verb', opposite, relation_direction) or
                            self._is_similar(word1.dep, word2.dep, pos, 'adjective', opposite, relation_direction) or
                            self._is_similar(word1.dep, word2.dep, pos, 'adverb', opposite, relation_direction)))):

                    relative_alignments.append((word1.index, word2.index))
                    word_similarities.append(word_relatedness_alignment_stanford(word1, word2, self.config)[0])

        alignment_results = {}

        for i, alignment in enumerate(relative_alignments):
            alignment_results[(alignment[0], alignment[1])] = word_similarities[i]

        return alignment_results

    def _find_dependency_difference(self, pos, source_word, target_word, alignments):
        compare_parents = self._compare_aligned_nodes(source_word.parents, target_word.parents, pos, False, 'parent', alignments)
        compare_children = self._compare_aligned_nodes(source_word.children, target_word.children, pos, False, 'child', alignments)
        compare_parent_children = self._compare_aligned_nodes(source_word.parents, target_word.children, pos, True, 'parent_child', alignments)
        compare_children_parent = self._compare_aligned_nodes(source_word.parents, target_word.children, pos, True, 'child_parent', alignments)

        labels_source = []
        labels_target = []

        children_matched_source = []
        parents_matched_source = []
        children_matched_target = []
        parents_matched_target = []

        for item in compare_children.keys():
            children_matched_target.append(item[1])
            children_matched_source.append(item[0])

        for item in compare_parents.keys():
            parents_matched_target.append(item[1])
            parents_matched_source.append(item[0])

        for item in compare_children_parent.keys():
            children_matched_target.append(item[1])
            parents_matched_target.append(item[1])
            children_matched_source.append(item[0])
            parents_matched_source.append(item[0])

        for item in compare_parent_children.keys():
            children_matched_target.append(item[1])
            parents_matched_target.append(item[1])
            children_matched_source.append(item[0])
            parents_matched_source.append(item[0])

        for item in target_word.children:
            if item.index not in children_matched_target:
                labels_target.append(item.dep)

        for item in target_word.parents:
            if item.index not in parents_matched_target:
                labels_target.append(item.dep)

        for item in source_word.children:
            if item.index not in children_matched_source:
                labels_source.append(item.dep)

        for item in source_word.parents:
            if item.index not in parents_matched_source:
                labels_source.append(item.dep)

        return [labels_source, labels_target]

    def _compile_pair_context_info(self, source_word, target_word, alignments):
        context_source_labels = []
        context_target_labels = []

        context_info = {}

        for item in source_word.children + source_word.parents:
            context_source_labels.append(item.dep)

        for item in target_word.children + target_word.parents:
            context_target_labels.append(item.dep)

        difference = self._find_dependency_difference(source_word.get_category(), source_word, target_word, alignments)
        context_info['srcDiff'] = difference[0]
        context_info['tgtDiff'] = difference[1]
        context_info['srcCon'] = context_source_labels
        context_info['tgtCon'] = context_target_labels

        return context_info

    def compile_context_info(self, source, target, alignments):
        context_info = []
        for a in alignments:
            context_info.append(self._compile_pair_context_info(source[a[0]-1], target[a[1]-1], alignments))

        return context_info
