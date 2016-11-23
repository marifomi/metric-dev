from alignment.aligner_config import AlignerConfig
from utils import load_resources
from alignment.util import *
from utils.word_sim import *
from utils.named_entity_group import NamedEntityGroup
from utils.core_nlp_utils import *

__author__ = 'anton'


class AlignerStanford(object):

    config = None
    alignments = set()
    source_indices_aligned = set()
    target_indices_aligned = set()

    def __init__(self, language):
        self.config = AlignerConfig(language)

        if "paraphrases" in self.config.selected_lexical_resources:
            load_resources.load_ppdb(self.config.path_to_ppdb)
        if "distributional" in self.config.selected_lexical_resources:
            load_resources.load_word_vectors(self.config.path_to_vectors)

    def _add_to_alignments(self, source_index, target_index):
        self.alignments.add((source_index, target_index))
        self.source_indices_aligned.add(source_index)
        self.target_indices_aligned.add(target_index)

    def _align_ending_punctuation(self, source, target):
        if (source[len(source) - 1].is_sentence_ending_punctuation() and target[len(target) - 1].is_sentence_ending_punctuation())\
                or source[len(source) - 1].form == target[len(target) - 1].form:
            self._add_to_alignments(len(source), len(target))
        elif source[len(source) - 2].is_sentence_ending_punctuation() and target[len(target) - 1].is_sentence_ending_punctuation():
            self._add_to_alignments(len(source) - 1, len(target))
        elif source[len(source) - 1].is_sentence_ending_punctuation() and target[len(target) - 2].is_sentence_ending_punctuation():
            self._add_to_alignments(len(source), len(target) - 1)
        elif source[len(source) - 2].is_sentence_ending_punctuation() and target[len(target) - 2].is_sentence_ending_punctuation():
            self._add_to_alignments(len(source) - 1, len(target) - 1)

        return

    def _align_contiguous_sublists(self, source, target):
        sublists = find_all_common_contiguous_sublists(source, target, True)

        for item in sublists:
            only_stopwords = True
            for jtem in item[0]:
                if jtem not in cobalt_stopwords and jtem not in punctuations:
                    only_stopwords = False
                    break
            if len(item[0]) >= 3 and not only_stopwords:
                for j in range(len(item[0])):
                    if item[0][j]+1 not in self.source_indices_aligned \
                            and item[1][j]+1 not in self.target_indices_aligned \
                            and (item[0][j]+1, item[1][j]+1) not in self.alignments:
                        self._add_to_alignments(item[0][j]+1, item[1][j]+1)

    def _align_hyphenated_word_groups(self, source, target):
        for word in source:
            if word.index in self.source_indices_aligned:
                continue
            if '-' in word.form and word.form != '-':
                tokens = word.form.split('-')
                sublists = find_all_common_contiguous_sublists(tokens, target)
                for item in sublists:
                    if len(item[1]) > 1:
                        for jtem in item[1]:
                            if (word.index, jtem+1) not in self.alignments:
                                self._add_to_alignments(word.index, jtem+1)

        for word in target:
            if word.index in self.target_indices_aligned:
                continue
            if '-' in word.form and word.form != '-':
                tokens = word.form.split('-')
                sublists = find_all_common_contiguous_sublists(source, tokens)
                for item in sublists:
                    if len(item[0]) > 1:
                        for jtem in item[0]:
                            if (jtem+1, word.index) not in self.alignments:
                                self._add_to_alignments(jtem+1, word.index)

    def _extend_named_entities_group_using_opposite_sentence(self, sentence, same_ne_groups, opposite_ne_groups):
        # learn from the other sentence that a certain word/phrase is a named entity
        for word in sentence:
            included = False
            for source_ne_group in same_ne_groups:
                if word.index in source_ne_group.indicies:
                    included = True
                    break
            if included or not word.form.isupper():
                continue
            for opposite_ne_group in opposite_ne_groups:
                if word.form in opposite_ne_group.forms:
                    new_ne_group = NamedEntityGroup([word.index], [word], opposite_ne_group.ner)

                    # check if the current item is part of a named entity part of which has already been added (by checking contiguousness)
                    is_part = False
                    for group in same_ne_groups:
                        if group.indicies[-1] == new_ne_group.indicies[0] - 1:
                            group.indicies.append(word.index)
                            group.words.append(word)
                            group.forms.append(word.form)
                            is_part = True
                    if not is_part:
                        same_ne_groups.append(new_ne_group)
                elif is_acronym_stanford(word, opposite_ne_group) \
                        and next((x for x in same_ne_groups if word.form in x.forms), None) is None:
                    same_ne_groups.append(NamedEntityGroup([word.index], [word], opposite_ne_group.ner))

    def _count_form_occurences(self, ne_group, groups):
        count = 0
        for group in groups:
            if group.forms == ne_group.forms:
                count += 1

        return count
    
    def _align_named_entity_subsets(self, smaller_group, bigger_group, source_is_smaller, full_smaller_words):
        unaligned_indicies_bigger = []
        unaligned_indicies_bigger.extend(bigger_group.indicies)

        for i, smaller_form in enumerate(smaller_group.forms):
            for j, bigger_form in enumerate(bigger_group.forms):
                if smaller_form == bigger_form:
                    if source_is_smaller:
                        self._add_to_alignments(smaller_group.indicies[i], bigger_group.indicies[j])
                    else:
                        self._add_to_alignments(bigger_group.indicies[j], smaller_group.indicies[i])

                    if bigger_group.indicies[j] in unaligned_indicies_bigger:
                        unaligned_indicies_bigger.remove(bigger_group.indicies[j])

        for i, smaller_word in enumerate(smaller_group.words):
            for j, bigger_word in enumerate(bigger_group.words):
                # find if the current term in the longer name has already been aligned, do not align_sentence it in that case
                dont_insert = smaller_word.is_punctuation() or bigger_word.is_punctuation()
                for a in self.alignments:
                    if (source_is_smaller and a[1] == bigger_group.indicies[j]) or (not source_is_smaller and a[0] == bigger_group.indicies[j]):
                        dont_insert = True
                        break
                for word in full_smaller_words:
                    if bigger_word.form == word.form:
                        dont_insert = True
                        break

                if bigger_group.indicies[j] not in unaligned_indicies_bigger or dont_insert:
                    continue

                if source_is_smaller:
                    self._add_to_alignments(smaller_group.indicies[i], bigger_group.indicies[j])
                else:
                    self._add_to_alignments(bigger_group.indicies[j], smaller_group.indicies[i])

    def _align_named_entities(self, source, target):
        source_ne_groups = sorted(find_named_entity_groups(source), key=NamedEntityGroup.sort_key)
        target_ne_groups = sorted(find_named_entity_groups(target), key=NamedEntityGroup.sort_key)

        self._extend_named_entities_group_using_opposite_sentence(source, source_ne_groups, target_ne_groups)
        self._extend_named_entities_group_using_opposite_sentence(target, target_ne_groups, source_ne_groups)

        if len(source_ne_groups) == 0 or len(target_ne_groups) == 0:
            return

        source_ne_groups_aligned = []
        target_ne_groups_aligned = []

        # align_sentence all full matches
        for source_group in source_ne_groups:
            if self._count_form_occurences(source_group, source_ne_groups) > 1:
                continue

            for target_group in target_ne_groups:
                if self._count_form_occurences(target_group, target_ne_groups) > 1:
                    continue

                # get rid of dots and hyphens
                canonical_source_forms = [i.replace('.', '').replace('-', '') for i in source_group.forms]
                canonical_target_forms = [j.replace('.', '').replace('-', '') for j in target_group.forms]

                if canonical_source_forms == canonical_target_forms:
                    for k in range(len(source_group.indicies)):
                        if (source_group.indicies[k], target_group.indicies[k]) not in self.alignments:
                            self._add_to_alignments(source_group.indicies[k], target_group.indicies[k])
                    source_ne_groups_aligned.append(source_group)
                    target_ne_groups_aligned.append(target_group)

        # align_sentence acronyms with their elaborations
        for source_group in source_ne_groups:
            for target_group in target_ne_groups:
                if len(source_group.words) == 1 and is_acronym_stanford(source_group.words[0], target_group):
                    for i in range(len(target_group.indicies)):
                        if (source_group.indicies[0], target_group.indicies[i]) not in self.alignments:
                            self._add_to_alignments(source_group.indicies[0], target_group.indicies[i])

                elif len(target_group.words) == 1 and is_acronym_stanford(target_group.words[0], source_group):
                    for i in range(len(source_group.indicies)):
                        if (source_group.indicies[i], target_group.indicies[0]) not in self.alignments:
                            self._add_to_alignments(source_group.indicies[i], target_group.indicies[0])


        # align_sentence subset matches
        for source_group in source_ne_groups:
            if source_group in source_ne_groups_aligned or self._count_form_occurences(source_group, source_ne_groups) > 1:
                continue

            for target_group in target_ne_groups:
                if target_group in target_ne_groups_aligned or source_group.ner != target_group.ner \
                        or self._count_form_occurences(target_group, target_ne_groups) > 1:
                    continue

                # find if the first is a part of the second
                if is_sublist(source_group.forms, target_group.forms):
                    self._align_named_entity_subsets(source_group, target_group, True, source)
                # else find if the second is a part of the first
                if is_sublist(target_group.forms, source_group.forms):
                    self._align_named_entity_subsets(target_group, source_group, False, target)

        return

    def _is_similar(self, dep1, dep2, pos1, pos2, is_opposite, relation):
        result = False
        group = self.config.get_similar_group(pos1, pos2, is_opposite, relation)

        if is_opposite:
            for subgroup in group:
                if dep1 in subgroup[0] and dep2 in subgroup[1]:
                    result = True
        else:
            for subgroup in group:
                if dep1 in subgroup and dep2 in subgroup:
                    result = True

        return result

    def _compare_nodes(self, source, target, pos, opposite, relation_direction):
        # search for nodes in common or equivalent function
        result = {}

        for word1 in source:
            for word2 in target:

                if ((word1.index, word2.index) in self.alignments or word_relatedness_alignment(word1, word2, self.config) >= self.config.alignment_similarity_threshold) and (
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

                    result[(word1.index, word2.index)] = word_relatedness_alignment(word1, word2, self.config)

        return result

    def _calculate_absolute_score(self, word_similarities):

        max_left = {}
        max_right = {}

        for similarity in word_similarities.keys():
            if not similarity[0] in max_left or word_similarities[max_left[similarity[0]]] < word_similarities[similarity]:
                max_left[similarity[0]] = similarity

            if not similarity[1] in max_right or word_similarities[max_right[similarity[1]]] < word_similarities[similarity]:
                max_right[similarity[1]] = similarity

        left_right = dict()
        left_right.update(max_left)
        left_right.update(max_right)

        max_relations = set(left_right.values())

        score = 0
        source_nodes_considered = []
        target_nodes_considered = []

        for rel in max_relations:

            if rel[0] not in source_nodes_considered and rel[1] not in target_nodes_considered:
                score += word_similarities[rel]
                source_nodes_considered.append(rel[0])
                target_nodes_considered.append(rel[1])

        return score

    def _find_dependency_similarity(self, pos, source_word, target_word):
        comparison = dict()
        comparison.update(self._compare_nodes(source_word.parents, target_word.parents, pos, False, 'parent'))
        comparison.update(self._compare_nodes(source_word.children, target_word.children, pos, False, 'child'))
        comparison.update(self._compare_nodes(source_word.parents, target_word.children, pos, True, 'parent_child'))
        comparison.update(self._compare_nodes(source_word.parents, target_word.children, pos, True, 'child_parent'))

        alignments = []
        word_similarities = {}

        for alignment in comparison.keys():
            alignments.append([alignment[0], alignment[1]])
            word_similarities[alignment] = comparison[alignment]

        return [self._calculate_absolute_score(word_similarities), alignments]

    def _align_on_dependency_match(self, pos, pos_code, source, target):
        pos_count_in_source = 0
        evidence_counts_matrix = {}
        relative_alignments_matrix = {}
        word_similarities = {}

        # construct the two matrices in the following loop
        for item in source:
            i = item.index

            if i in self.source_indices_aligned or not item.matches_pos_code(pos_code):
                continue

            pos_count_in_source += 1

            for jtem in target:
                j = jtem.index

                if j in self.target_indices_aligned or not jtem.matches_pos_code(pos_code):
                    continue

                if word_relatedness_alignment(item, jtem, self.config) < self.config.alignment_similarity_threshold:
                    continue

                word_similarities[(i, j)] = word_relatedness_alignment(item, jtem, self.config)

                dependency_similarity = self._find_dependency_similarity(pos, item, jtem)

                if word_similarities[(i, j)] == self.config.alignment_similarity_threshold:
                    if word_similarities[(i, j)] + dependency_similarity[0] <= 1.0:
                        continue

                if dependency_similarity[0] >= self.config.alignment_similarity_threshold:
                    evidence_counts_matrix[(i, j)] = dependency_similarity[0]
                    relative_alignments_matrix[(i, j)] = dependency_similarity[1]
                else:
                    evidence_counts_matrix[(i, j)] = 0

        # now use the collected stats to align_sentence
        for n in range(pos_count_in_source):

            max_overall_value_for_pass = 0
            index_pair_with_strongest_tie_for_pass = [-1, -1]

            for item in source:
                i = item.index

                if i in self.source_indices_aligned or not item.matches_pos_code(pos_code):
                    continue

                for jtem in target:
                    j = jtem.index
                    if j in self.target_indices_aligned or not jtem.matches_pos_code(pos_code):
                        continue

                    if (i, j) in evidence_counts_matrix and self.config.theta * word_similarities[(i, j)] + (1 - self.config.theta) * evidence_counts_matrix[(i, j)] > max_overall_value_for_pass:
                        max_overall_value_for_pass = self.config.theta * word_similarities[(i, j)] + (1 - self.config.theta) * evidence_counts_matrix[(i, j)]
                        index_pair_with_strongest_tie_for_pass = [i, j]

            if max_overall_value_for_pass > 0:
                self._add_to_alignments(index_pair_with_strongest_tie_for_pass[0], index_pair_with_strongest_tie_for_pass[1])
            else:
                break

        return

    def _get_unaligned_words(self, source, target, word_function):
        content_source = []
        content_target = []

        for word in source:
            if word.index in self.source_indices_aligned or not word_function(word):
                continue
            content_source.append(word)

        for word in target:
            if word.index in self.target_indices_aligned or not word_function(word):
                continue
            content_target.append(word)

        return content_source, content_target

    def _collect_evidence_from_textual_neighborhood(self, unaligned_source, unaligned_target, full_source, full_target):
        word_similarities = {}

        textual_neighborhood_similarities = {}

        for source_word in unaligned_source:
            for target_word in unaligned_target:

                word_similarities[(source_word.index, target_word.index)] = word_relatedness_alignment(source_word, target_word, self.config)

                # textual neighborhood similarities
                source_neighborhood = find_textual_neighborhood_stanford(full_source, source_word.index, 3, 3)
                target_neighborhood = find_textual_neighborhood_stanford(full_target, target_word.index, 3, 3)
                evidence = 0

                for source_neighbor in source_neighborhood:
                    for target_neighbor in target_neighborhood:
                        if (source_neighbor.index, target_neighbor.index) in self.alignments \
                                or word_relatedness_alignment(source_neighbor, target_neighbor, self.config) >= self.config.alignment_similarity_threshold:
                            evidence += word_relatedness_alignment(source_neighbor, target_neighbor, self.config)

                textual_neighborhood_similarities[(source_word.index, target_word.index)] = evidence

        return word_similarities, textual_neighborhood_similarities

    def _collect_evidence_from_dependency_neighborhood(self, unaligned_source, unaligned_target):
        word_similarities = {}
        dependency_neighborhood_similarities = {}

        for source_word in unaligned_source:
            for target_word in unaligned_target:
                i = source_word.index
                j = target_word.index

                if (source_word.lemma != target_word.lemma) and (word_relatedness_alignment(source_word, target_word, self.config) < self.config.alignment_similarity_threshold):
                    word_similarities[(i, j)] = 0
                    dependency_neighborhood_similarities[(i, j)] = 0
                    continue

                word_similarities[(i, j)] = word_relatedness_alignment(source_word, target_word, self.config)

                evidence = 0
                for source_parent in source_word.parents:
                    for target_parent in target_word.parents:
                        if (source_parent.index, target_parent.index) in self.alignments:
                            evidence += 1
                for source_child in source_word.children:
                    for target_child in target_word.children:
                        if (source_child.index, target_child.index) in self.alignments:
                            evidence += 1

                dependency_neighborhood_similarities[(i, j)] = evidence

        return word_similarities, dependency_neighborhood_similarities

    def _collect_evidence_from_textual_neighborhood_for_stopwords(self, unaligned_source, unaligned_target, full_source, full_target):
        word_similarities = {}
        textual_neighborhood_similarities = {}

        for source_word in unaligned_source:
            for target_word in unaligned_target:
                i = source_word.index
                j = target_word.index

                if word_relatedness_alignment(source_word, target_word, self.config) < self.config.alignment_similarity_threshold:
                    word_similarities[(i, j)] = 0
                    textual_neighborhood_similarities[(i, j)] = 0
                    continue

                word_similarities[(i, j)] = word_relatedness_alignment(source_word, target_word, self.config)

                # textual neighborhood evidence, increasing evidence if content words around this stop word are aligned
                evidence = 0

                k = i
                l = j
                while k > 0:
                    if full_source[k-1].is_stopword() or full_source[k-1].is_punctuation():
                        k -= 1
                    else:
                        break
                while l > 0:
                    if full_target[l-1].is_stopword() or full_target[l-1].is_punctuation():
                        l -= 1
                    else:
                        break

                m = i
                n = j

                while m < len(full_source) - 1:
                    if full_source[m-1].is_stopword() or full_source[m-1].is_punctuation():
                        m += 1
                    else:
                        break
                while n < len(full_target) - 1:
                    if full_target[n-1].is_stopword() or full_target[n-1].is_punctuation():
                        n += 1
                    else:
                        break

                if (k, l) in self.alignments:
                    evidence += 1

                if (m, n) in self.alignments:
                    evidence += 1

                textual_neighborhood_similarities[(i, j)] = evidence

        return word_similarities, textual_neighborhood_similarities

    def _align_words(self, unaligned_source, unaligned_target, word_similarities, neighborhood_similarities, limit_by_neighborhood):
        # now align_sentence: find the best alignment in each iteration of the following loop and include in alignments if good enough
        for item in range(len(unaligned_source)):
            highest_weighted_similarity = 0
            best_word_similarity = 0
            best_neighborhood_similarity = 0
            best_source = None
            best_target = None

            for source_word in unaligned_source:
                for target_word in unaligned_target:
                    i = source_word.index
                    j = target_word.index

                    if (i, j) not in word_similarities:
                        continue

                    if word_similarities[(i, j)] == self.config.alignment_similarity_threshold:
                        if word_similarities[(i, j)] + neighborhood_similarities[(i, j)] <= 1.0:
                            continue

                    if self.config.theta * word_similarities[(i, j)] + (1 - self.config.theta) * neighborhood_similarities[(i, j)] > highest_weighted_similarity:
                        highest_weighted_similarity = self.config.theta * word_similarities[(i, j)] + (1 - self.config.theta) * neighborhood_similarities[(i, j)]
                        best_source = source_word
                        best_target = target_word
                        best_word_similarity = word_similarities[(i, j)]
                        best_neighborhood_similarity = neighborhood_similarities[(i, j)]

            if best_word_similarity >= self.config.alignment_similarity_threshold \
                    and (not limit_by_neighborhood or best_neighborhood_similarity > 0) \
                    and best_source.index not in self.source_indices_aligned \
                    and best_target.index not in self.target_indices_aligned:
                self._add_to_alignments(best_source.index, best_target.index)
                if best_source is not None:
                    unaligned_source.remove(best_source)
                if best_target is not None:
                    unaligned_target.remove(best_target)

        return unaligned_source, unaligned_target

    def _align_remaining_if_hyphenated(self, remaining_source, remaining_target, full_source, full_target):
         # look if any remaining word is a part of a hyphenated word
        for source_word in remaining_source:
            if '-' in source_word.form and source_word.form != '-':
                tokens = source_word.form.split('-')
                for item in find_all_common_contiguous_sublists(tokens, full_target):
                    if len(item[0]) == 1 and full_target[item[1][0]].is_stopword():
                        for jtem in item[1]:
                            if (source_word.index, jtem+1) not in self.alignments and jtem+1 not in self.target_indices_aligned:
                                self._add_to_alignments(source_word.index, jtem+1)

        for target_word in remaining_target:
            if '-' in target_word.form and target_word.form != '-':
                tokens = target_word.form.split('-')
                for item in find_all_common_contiguous_sublists(full_source, tokens):
                    if len(item[0]) == 1 and not full_source[item[0][0]].is_stopword():
                        for jtem in item[0]:
                            if (jtem+1, target_word.index) not in self.alignments and target_word.index not in self.target_indices_aligned:
                                self._add_to_alignments(jtem+1, target_word.index)

    def _align_content_words(self, source, target):
        def is_content_word(word):
            return word.is_content_word()

        content_source, content_target = self._get_unaligned_words(source, target, is_content_word)
        word_similarities, neighborhood_similarities = self._collect_evidence_from_textual_neighborhood(content_source, content_target, source, target)
        remaining_source, remaining_target = self._align_words(content_source, content_target, word_similarities, neighborhood_similarities, False)
        self._align_remaining_if_hyphenated(remaining_source, remaining_target, source, target)

    def _align_stop_words_by_dependency_neighborhood(self, source, target):
        def is_stopword(word):
            return word.is_stopword()

        stop_source, stop_target = self._get_unaligned_words(source, target, is_stopword)
        word_similarities, neighborhood_similarities = self._collect_evidence_from_dependency_neighborhood(stop_source, stop_target)
        self._align_words(stop_source, stop_target, word_similarities, neighborhood_similarities, True)

    def _align_stop_words_and_punctuations_by_textual_neighborhood(self, source, target):
        def is_stopword_or_punctuation(word):
            return word.is_stopword() or word.is_punctuation()

        stop_source, stop_target = self._get_unaligned_words(source, target, is_stopword_or_punctuation)
        word_similarities, neighborhood_similarities = self._collect_evidence_from_textual_neighborhood_for_stopwords(stop_source, stop_target, source, target)
        self._align_words(stop_source, stop_target, word_similarities, neighborhood_similarities, True)

    def align(self, source, target):
        self.alignments = set()
        self.source_indices_aligned = set()
        self.target_indices_aligned = set()

        self._align_ending_punctuation(source, target)

        self._align_contiguous_sublists(source, target)

        self._align_hyphenated_word_groups(source, target)

        self._align_named_entities(source, target)

        self._align_on_dependency_match('verb', 'v', source, target)
        self._align_on_dependency_match('noun', 'n', source, target)
        self._align_on_dependency_match('adjective', 'j', source, target)
        self._align_on_dependency_match('adverb', 'r', source, target)

        self._align_content_words(source, target)

        self._align_stop_words_by_dependency_neighborhood(source, target)

        self._align_stop_words_and_punctuations_by_textual_neighborhood(source, target)

        return self.alignments

