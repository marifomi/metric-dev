import os

from configparser import ConfigParser
from json import loads


class ContextDifference(object):

    left_context = []
    right_context = []
    left_difference = []
    right_difference = []


class ContextEvidence(object):

    config = ConfigParser()
    _equivalent_functions = dict()

    def __init__(self):
        self.config.readfp(open('config/equivalent_dependencies.cfg'))

    def get_similar_group(self, left_category, right_category, is_opposite, relation):
        group_name = left_category + '_' + ('opposite_' if is_opposite else '') + right_category + '_' + relation

        if group_name in self._equivalent_functions:
            return self._equivalent_functions[group_name]

        similar_group = []
        for line in self.config.get('Equivalent Functions', group_name).splitlines():
            similar_group.append(loads(line.strip()))

        self._equivalent_functions[group_name] = similar_group

        return similar_group

    def is_similar(self, left_pos, right_pos, context_category, target_category, is_opposite, relation):

        result = False
        group = self.get_similar_group(context_category, target_category, is_opposite, relation)

        if is_opposite:
            for subgroup in group:
                if left_pos in subgroup[0] and right_pos in subgroup[1]:
                    result = True
        else:
            for subgroup in group:
                if left_pos in subgroup and right_pos in subgroup:
                    result = True

        return result

    def equivalent_context(self, right_target_word, left_context_words, right_context_words, relation, opposite, alignments):

        """ Alignments: list of tuples (left idx, right idx) idx starts with 1"""

        equivalent_context_words = {}
        for left_context_word in left_context_words:
            for right_context_word in right_context_words:
                if (left_context_word.index, right_context_word.index) not in alignments + [(0, 0)]: # root word indexes
                    continue
                if left_context_word.dep == right_context_word.dep:
                    equivalent_context_words[left_context_word.index] = right_context_word.index
                    continue
                if len(left_context_word.get_category()) == 0 or len(right_context_word.get_category()) == 0:
                    continue
                first = left_context_word
                second = right_context_word
                if relation == 'child_parent':
                    first = right_context_word
                    second = left_context_word
                if self.is_similar(first.pos, second.pos, left_context_word.get_category(), right_target_word.get_category(), opposite, relation):
                    equivalent_context_words[left_context_word.index] = right_context_word.index

        return equivalent_context_words

    def context_differences(self, left_word, right_word, left_parse, right_parse, alignments):

        left_head = left_parse[left_word.head]
        right_head = right_parse[right_word.head]
        left_children = [left_parse[x - 1] for x in left_word.find_children_nodes(left_parse)]
        right_children = [right_parse[x - 1] for x in right_word.find_children_nodes(right_parse)]

        equivalent_context = {}
        equivalent_context.update(self.equivalent_context(right_word, [left_head], [right_head], 'parent', False, alignments))
        equivalent_context.update(self.equivalent_context(right_word, left_children, right_children, 'child', False, alignments))
        equivalent_context.update(self.equivalent_context(right_word, [left_head], right_children, 'child_parent', True, alignments))
        equivalent_context.update(self.equivalent_context(right_word, [left_head], right_children, 'parent_child', True, alignments))

        different_left_dependencies = []
        different_right_dependencies = []
        for word in [left_head] + left_children:
            if word.index not in equivalent_context.keys():
                different_left_dependencies.append(word.dep)
        for word in [right_head] + right_children:
            if word.index not in equivalent_context.values():
                different_right_dependencies.append(word.dep)

        return different_left_dependencies, different_right_dependencies