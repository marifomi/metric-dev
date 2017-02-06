import codecs
import os

from alignment.aligner_config import AlignerConfig
from alignment.util import *
from utils.word_sim import *
from utils.core_nlp_utils import *
from utils import load_resources
from operator import itemgetter
from alignment.context_info_compiler import ContextInfoCompiler
from utils.stanford_format import StanfordParseLoader


class Aligner(object):

    config = None

    def __init__(self, language):
        self.config = AlignerConfig(language)
        self.alignments = []

    def is_similar(self, item1, item2, pos1, pos2, is_opposite, relation):
        result = False
        group = self.config.get_similar_group(pos1, pos2, is_opposite, relation)

        if is_opposite:
            for subgroup in group:
                if item1 in subgroup[0] and item2 in subgroup[1]:
                    result = True
        else:
            for subgroup in group:
                if item1 in subgroup and item2 in subgroup:
                    result = True

        return result

    def compareNodes(self, sourceNodes, targetNodes, pos, opposite, relationDirection, existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas):
        # search for nodes in common or equivalent function

        relativeAlignments = []
        wordSimilarities = []

        for ktem in sourceNodes:
            for ltem in targetNodes:

                # To compare root nodes with dummy index 0:
                if ktem[0] == 0:
                    word1 = Word(ktem[0], ktem[1])
                    word1.lemma = 'ROOT'
                    word1.pos = 'root'
                    word1.dep = 'root'
                else:
                    word1 = Word(ktem[0], ktem[1])
                    word1.lemma = sourceLemmas[ktem[0]-1]
                    word1.pos = sourcePosTags[ktem[0]-1]
                    word1.dep = ktem[2]

                if ltem[0] == 0:
                    word2 = Word(ltem[0], ltem[1])
                    word2.lemma = 'ROOT'
                    word2.pos = 'root'
                    word2.dep = 'root'
                else:
                    word2 = Word(ltem[0], ltem[1])
                    word2.lemma = targetLemmas[ltem[0]-1]
                    word2.pos = targetPosTags[ltem[0]-1]
                    word2.dep = ltem[2]

                if ([ktem[0], ltem[0]] in existingAlignments or word_relatedness_alignment(word1, word2, self.config) >= self.config.alignment_similarity_threshold) and (
                    (ktem[2] == ltem[2]) or
                        ((pos != '' and relationDirection != 'child_parent') and (
                            self.is_similar(ktem[2], ltem[2], pos, 'noun', opposite, relationDirection) or
                            self.is_similar(ktem[2], ltem[2], pos, 'verb', opposite, relationDirection) or
                            self.is_similar(ktem[2], ltem[2], pos, 'adjective', opposite, relationDirection) or
                            self.is_similar(ktem[2], ltem[2], pos, 'adverb', opposite, relationDirection))) or
                        ((pos != '' and relationDirection == 'child_parent') and (
                            self.is_similar(ltem[2], ktem[2], pos, 'noun', opposite, relationDirection) or
                            self.is_similar(ltem[2], ktem[2], pos, 'verb', opposite, relationDirection) or
                            self.is_similar(ltem[2], ktem[2], pos, 'adjective', opposite, relationDirection) or
                            self.is_similar(ltem[2], ktem[2], pos, 'adverb', opposite, relationDirection)))):

                    relativeAlignments.append([ktem[0], ltem[0]])
                    wordSimilarities.append(word_relatedness_alignment(word1, word2, self.config))

        alignmentResults = {}

        for i, alignment in enumerate(relativeAlignments):
            alignmentResults[(alignment[0], alignment[1])] = wordSimilarities[i]

        return alignmentResults

    def compareNodesScoring(self, sourceNodes, targetNodes, pos, opposite, relationDirection, existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas):
        # search for nodes in common or equivalent function

        relativeAlignments = []
        wordSimilarities = []

        for ktem in sourceNodes:
            for ltem in targetNodes:

                word1 = Word(ktem[0], ktem[1])
                word1.lemma = sourceLemmas[ktem[0]-1]
                word1.pos = sourcePosTags[ktem[0]-1]
                word1.dep = ktem[2]
                word2 = Word(ltem[0], ltem[1])
                word2.lemma = targetLemmas[ltem[0]-1]
                word2.pos = targetPosTags[ltem[0]-1]
                word2.dep = ltem[2]

                if ([ktem[0], ltem[0]] in existingAlignments or (ktem[0] == 0 and ltem[0] == 0)) and (
                    (ktem[2] == ltem[2]) or
                        ((pos != '' and relationDirection != 'child_parent') and (
                            self.is_similar(ktem[2], ltem[2], pos, 'noun', opposite, relationDirection) or
                            self.is_similar(ktem[2], ltem[2], pos, 'verb', opposite, relationDirection) or
                            self.is_similar(ktem[2], ltem[2], pos, 'adjective', opposite, relationDirection) or
                            self.is_similar(ktem[2], ltem[2], pos, 'adverb', opposite, relationDirection))) or
                        ((pos != '' and relationDirection == 'child_parent') and (
                            self.is_similar(ltem[2], ktem[2], pos, 'noun', opposite, relationDirection) or
                            self.is_similar(ltem[2], ktem[2], pos, 'verb', opposite, relationDirection) or
                            self.is_similar(ltem[2], ktem[2], pos, 'adjective', opposite, relationDirection) or
                            self.is_similar(ltem[2], ktem[2], pos, 'adverb', opposite, relationDirection)))):

                    relativeAlignments.append([ktem[0], ltem[0]])
                    wordSimilarities.append(word_relatedness_alignment(word1, word2, self.config))

        alignmentResults = {}

        for i, alignment in enumerate(relativeAlignments):
            alignmentResults[(alignment[0], alignment[1])] = wordSimilarities[i]

        return alignmentResults

    def calculateAbsoluteScore(self, wordSimilarities):

        maxLeft = {}
        maxRight = {}

        maxLeftList = {}
        maxRightList = {}

        for similarity in wordSimilarities.keys():
            if not similarity[0] in maxLeft or wordSimilarities[maxLeft[similarity[0]]] < wordSimilarities[similarity]:
                maxLeft[similarity[0]] = similarity
                maxLeftList[similarity[0]] = similarity[1]

            if not similarity[1] in maxRight or wordSimilarities[maxRight[similarity[1]]] < wordSimilarities[similarity]:
                maxRight[similarity[1]] = similarity
                maxRightList[similarity[1]] = similarity[0]

        leftRight = dict()
        leftRight.update(maxLeft)
        leftRight.update(maxRight)

        maxRelations = set(leftRight.values())

        score = 0
        sourceNodesConsidered = []
        targetNodesConsidered = []

        for rel in maxRelations:

            if rel[0] not in sourceNodesConsidered and rel[1] not in targetNodesConsidered:
                score += wordSimilarities[rel]
                sourceNodesConsidered.append(rel[0])
                targetNodesConsidered.append(rel[1])

        return score


    def findDependencySimilarity(self, pos, source, sourceIndex, target, targetIndex, sourceDParse, targetDParse, existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas):

        sourceWordParents = findParents(sourceDParse, sourceIndex, source)
        sourceWordChildren = findChildren(sourceDParse, sourceIndex, source)
        targetWordParents = findParents(targetDParse, targetIndex, target)
        targetWordChildren = findChildren(targetDParse, targetIndex, target)

        compareParents = self.compareNodes(sourceWordParents, targetWordParents, pos, False, 'parent', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)
        compareChildren = self.compareNodes(sourceWordChildren, targetWordChildren, pos, False, 'child', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)
        compareParentChildren = self.compareNodes(sourceWordParents, targetWordChildren, pos, True, 'parent_child', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)
        compareChildrenParent = self.compareNodes(sourceWordParents, targetWordChildren, pos, True, 'child_parent', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)

        comparisonResult = dict()
        comparisonResult.update(compareParents)
        comparisonResult.update(compareChildren)
        comparisonResult.update(compareChildrenParent)
        comparisonResult.update(compareParentChildren)

        alignments = []
        wordSimilarities = {}

        for alignment in comparisonResult.keys():
            alignments.append([alignment[0], alignment[1]])
            wordSimilarities[alignment] = comparisonResult[alignment]

        return [self.calculateAbsoluteScore(wordSimilarities),
                alignments]

    def findDependencyDifference(self, pos, source, sourceIndex, target, targetIndex, sourceDParse, targetDParse, existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas):

        sourceWordParents = findParents(sourceDParse, sourceIndex, source)
        sourceWordChildren = findChildren(sourceDParse, sourceIndex, source)
        targetWordParents = findParents(targetDParse, targetIndex, target)
        targetWordChildren = findChildren(targetDParse, targetIndex, target)

        compareParents = self.compareNodesScoring(sourceWordParents, targetWordParents, pos, False, 'parent', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)
        compareChildren = self.compareNodesScoring(sourceWordChildren, targetWordChildren, pos, False, 'child', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)
        compareParentChildren = self.compareNodesScoring(sourceWordParents, targetWordChildren, pos, True, 'parent_child', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)
        compareChildrenParent = self.compareNodesScoring(sourceWordParents, targetWordChildren, pos, True, 'child_parent', existingAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)

        labelsSource = []
        labelsTarget = []

        childrenMatchedSource = []
        parentsMatchedSource = []
        childrenMatchedTarget = []
        parentsMatchedTarget = []

        for item in compareChildren.keys():
            childrenMatchedTarget.append(item[1])
            childrenMatchedSource.append(item[0])

        for item in compareParents.keys():
            parentsMatchedTarget.append(item[1])
            parentsMatchedSource.append(item[0])

        for item in compareChildrenParent.keys():
            childrenMatchedTarget.append(item[1])
            parentsMatchedTarget.append(item[1])
            childrenMatchedSource.append(item[0])
            parentsMatchedSource.append(item[0])

        for item in compareParentChildren.keys():
            childrenMatchedTarget.append(item[1])
            parentsMatchedTarget.append(item[1])
            childrenMatchedSource.append(item[0])
            parentsMatchedSource.append(item[0])

        for item in targetWordChildren:
            if item[0] not in childrenMatchedTarget:
                labelsTarget.append(item[2])

        for item in targetWordParents:
            if item[0] not in parentsMatchedTarget:
                labelsTarget.append(item[2])

        for item in sourceWordChildren:
            if item[0] not in childrenMatchedSource:
                labelsSource.append(item[2])

        for item in sourceWordParents:
            if item[0] not in parentsMatchedSource:
                labelsSource.append(item[2])

        return [labelsSource, labelsTarget]

    ##############################################################################################################################
    def alignPos(self, pos, posCode, source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

        global scorer

        posAlignments = []

        sourceWordIndices = [i+1 for i in range(len(source))]
        targetWordIndices = [i+1 for i in range(len(target))]

        sourceWordIndicesAlreadyAligned = sorted(list(set([item[0] for item in existingAlignments])))
        targetWordIndicesAlreadyAligned = sorted(list(set([item[1] for item in existingAlignments])))

        sourceWords = [item[2] for item in source]
        targetWords = [item[2] for item in target]

        sourceLemmas = [item[3] for item in source]
        targetLemmas = [item[3] for item in target]

        sourcePosTags = [item[4] for item in source]
        targetPosTags = [item[4] for item in target]

        sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
        targetDParse = dependencyParseAndPutOffsets(targetParseResult)


        numberOfPosWordsInSource = 0

        evidenceCountsMatrix = {}
        relativeAlignmentsMatrix = {}
        wordSimilarities = {}

        # construct the two matrices in the following loop
        for i in sourceWordIndices:
            if i in sourceWordIndicesAlreadyAligned or (sourcePosTags[i-1][0].lower() != posCode and sourcePosTags[i-1].lower() != 'prp'):
                continue

            numberOfPosWordsInSource += 1

            for j in targetWordIndices:
                if j in targetWordIndicesAlreadyAligned or (targetPosTags[j-1][0].lower() != posCode and targetPosTags[j-1].lower() != 'prp'):
                    continue

                word1 = Word(i, sourceWords[i-1])
                word1.lemma = sourceLemmas[i-1]
                word1.pos = sourcePosTags[i-1]
                word2 = Word(j, targetWords[j-1])
                word2.lemma = targetLemmas[j-1]
                word2.pos = targetPosTags[j-1]

                if word_relatedness_alignment(word1, word2, self.config) < self.config.alignment_similarity_threshold:
                    continue

                wordSimilarities[(i, j)] = word_relatedness_alignment(word1, word2, self.config)

                dependencySimilarity = self.findDependencySimilarity(pos, source, i, target, j, sourceDParse, targetDParse, existingAlignments + posAlignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)

                if wordSimilarities[(i, j)] == self.config.alignment_similarity_threshold:
                    if wordSimilarities[(i, j)] + dependencySimilarity[0] <= 1.0:
                        continue

                if dependencySimilarity[0] >= self.config.alignment_similarity_threshold:
                    evidenceCountsMatrix[(i, j)] = dependencySimilarity[0]
                    relativeAlignmentsMatrix[(i, j)] = dependencySimilarity[1]
                else:
                    evidenceCountsMatrix[(i, j)] = 0

        # now use the collected stats to align_sentence
        for n in range(numberOfPosWordsInSource):

            maxEvidenceCountForCurrentPass = 0
            maxOverallValueForCurrentPass = 0
            indexPairWithStrongestTieForCurrentPass = [-1, -1]

            for i in sourceWordIndices:
                if i in sourceWordIndicesAlreadyAligned or sourcePosTags[i-1][0].lower() != posCode or sourceLemmas[i-1].lower() in cobalt_stopwords:
                    continue

                for j in targetWordIndices:
                    if j in targetWordIndicesAlreadyAligned or targetPosTags[j-1][0].lower() != posCode or targetLemmas[j-1].lower() in cobalt_stopwords:
                        continue

                    if (i, j) in evidenceCountsMatrix and self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * evidenceCountsMatrix[(i, j)] > maxOverallValueForCurrentPass:
                        maxOverallValueForCurrentPass = self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * evidenceCountsMatrix[(i, j)]
                        maxEvidenceCountForCurrentPass = evidenceCountsMatrix[(i, j)]
                        indexPairWithStrongestTieForCurrentPass = [i, j]

            #if maxEvidenceCountForCurrentPass > 0:
            if maxOverallValueForCurrentPass > 0:
                posAlignments.append(indexPairWithStrongestTieForCurrentPass)
                sourceWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[0])
                targetWordIndicesAlreadyAligned.append(indexPairWithStrongestTieForCurrentPass[1])

            else:
                break

        return posAlignments


    ##############################################################################################################################
    def alignNamedEntities(self, source, target, sourceParseResult, targetParseResult, existingAlignments):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

        global punctuations

        alignments = []

        sourceNamedEntities = ner(sourceParseResult)
        sourceNamedEntities = sorted(sourceNamedEntities, key=len)

        targetNamedEntities = ner(targetParseResult)
        targetNamedEntities = sorted(targetNamedEntities, key=len)


        # learn from the other sentence that a certain word/phrase is a named entity (learn for source from target)
        for item in source:
            alreadyIncluded = False
            for jtem in sourceNamedEntities:
                if item[1] in jtem[1]:
                    alreadyIncluded = True
                    break
            if alreadyIncluded or (len(item[2]) > 0 and not item[2][0].isupper()):
                continue
            for jtem in targetNamedEntities:
                if item[2] in jtem[2]:
                    # construct the item
                    newItem = [[item[0]], [item[1]], [item[2]], jtem[3]]

                    # check if the current item is part of a named entity part of which has already been added (by checking contiguousness)
                    partOfABiggerName = False
                    for k in range(len(sourceNamedEntities)):
                        if sourceNamedEntities[k][1][len(sourceNamedEntities[k][1])-1] == newItem[1][0] - 1:
                            sourceNamedEntities[k][0].append(newItem[0][0])
                            sourceNamedEntities[k][1].append(newItem[1][0])
                            sourceNamedEntities[k][2].append(newItem[2][0])
                            partOfABiggerName = True
                    if not partOfABiggerName:
                        sourceNamedEntities.append(newItem)
                elif is_acronym(item[2], jtem[2]) and [[item[0]], [item[1]], [item[2]], jtem[3]] not in sourceNamedEntities:
                    sourceNamedEntities.append([[item[0]], [item[1]], [item[2]], jtem[3]])



        # learn from the other sentence that a certain word/phrase is a named entity (learn for target from source)
        for item in target:
            alreadyIncluded = False
            for jtem in targetNamedEntities:
                if item[1] in jtem[1]:
                    alreadyIncluded = True
                    break
            if alreadyIncluded or (len(item[2]) > 0 and not item[2][0].isupper()):
                continue
            for jtem in sourceNamedEntities:
                if item[2] in jtem[2]:
                    # construct the item
                    newItem = [[item[0]], [item[1]], [item[2]], jtem[3]]

                    # check if the current item is part of a named entity part of which has already been added (by checking contiguousness)
                    partOfABiggerName = False
                    for k in range(len(targetNamedEntities)):
                        if targetNamedEntities[k][1][len(targetNamedEntities[k][1])-1] == newItem[1][0] - 1:
                            targetNamedEntities[k][0].append(newItem[0][0])
                            targetNamedEntities[k][1].append(newItem[1][0])
                            targetNamedEntities[k][2].append(newItem[2][0])
                            partOfABiggerName = True
                    if not partOfABiggerName:
                        targetNamedEntities.append(newItem)
                elif is_acronym(item[2], jtem[2]) and [[item[0]], [item[1]], [item[2]], jtem[3]] not in targetNamedEntities:
                    targetNamedEntities.append([[item[0]], [item[1]], [item[2]], jtem[3]])


        sourceWords = []
        targetWords = []

        for item in sourceNamedEntities:
            for jtem in item[1]:
                if item[3] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                    sourceWords.append(source[jtem-1][2])
        for item in targetNamedEntities:
            for jtem in item[1]:
                if item[3] in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                    targetWords.append(target[jtem-1][2])



        if len(sourceNamedEntities) == 0 or len(targetNamedEntities) == 0:
            return []




        sourceNamedEntitiesAlreadyAligned = []
        targetNamedEntitiesAlreadyAligned = []


        # align_sentence all full matches
        for item in sourceNamedEntities:
            if item[3] not in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                continue


            # do not align_sentence if the current source entity is present more than once
            count = 0
            for ktem in sourceNamedEntities:
                if ktem[2] == item[2]:
                    count += 1
            if count > 1:
                continue


            for jtem in targetNamedEntities:
                if jtem[3] not in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                    continue

                # do not align_sentence if the current target entity is present more than once
                count = 0
                for ktem in targetNamedEntities:
                    if ktem[2] == jtem[2]:
                        count += 1
                if count > 1:
                    continue


                # get rid of dots and hyphens
                canonicalItemWord = [i.replace('.', '') for i in item[2]]
                canonicalItemWord = [i.replace('-', '') for i in item[2]]
                canonicalJtemWord = [j.replace('.', '') for j in jtem[2]]
                canonicalJtemWord = [j.replace('-', '') for j in jtem[2]]

                if canonicalItemWord == canonicalJtemWord:
                    for k in range(len(item[1])):
                        if ([item[1][k], jtem[1][k]]) not in alignments:
                            alignments.append([item[1][k], jtem[1][k]])
                    sourceNamedEntitiesAlreadyAligned.append(item)
                    targetNamedEntitiesAlreadyAligned.append(jtem)

        # align_sentence acronyms with their elaborations
        for item in sourceNamedEntities:
            if item[3] not in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                continue
            for jtem in targetNamedEntities:
                if jtem[3] not in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                    continue

                if len(item[2])==1 and is_acronym(item[2][0], jtem[2]):
                    for i in range(len(jtem[1])):
                        if [item[1][0], jtem[1][i]] not in alignments:
                            alignments.append([item[1][0], jtem[1][i]])
                            sourceNamedEntitiesAlreadyAligned.append(item[1][0])
                            targetNamedEntitiesAlreadyAligned.append(jtem[1][i])

                elif len(jtem[2])==1 and is_acronym(jtem[2][0], item[2]):
                    for i in range(len(item[1])):
                        if [item[1][i], jtem[1][0]] not in alignments:
                            alignments.append([item[1][i], jtem[1][0]])
                            sourceNamedEntitiesAlreadyAligned.append(item[1][i])
                            targetNamedEntitiesAlreadyAligned.append(jtem[1][0])


        # align_sentence subset matches
        for item in sourceNamedEntities:
            if item[3] not in ['PERSON', 'ORGANIZATION', 'LOCATION'] or item in sourceNamedEntitiesAlreadyAligned:
                continue

            # do not align_sentence if the current source entity is present more than once
            count = 0
            for ktem in sourceNamedEntities:
                if ktem[2] == item[2]:
                    count += 1
            if count > 1:
                continue


            for jtem in targetNamedEntities:
                if jtem[3] not in ['PERSON', 'ORGANIZATION', 'LOCATION'] or jtem in targetNamedEntitiesAlreadyAligned:
                    continue

                if item[3] != jtem[3]:
                    continue

                # do not align_sentence if the current target entity is present more than once
                count = 0
                for ktem in targetNamedEntities:
                    if ktem[2] == jtem[2]:
                        count += 1
                if count > 1:
                    continue


                # find if the first is a part of the second
                if is_sublist(item[2], jtem[2]):
                    unalignedWordIndicesInTheLongerName = []
                    for ktem in jtem[1]:
                        unalignedWordIndicesInTheLongerName.append(ktem)
                    for k in range(len(item[2])):
                        for l in range(len(jtem[2])):
                            if item[2][k] == jtem[2][l] and [item[1][k], jtem[1][l]] not in alignments:
                                alignments.append([item[1][k], jtem[1][l]])
                                if jtem[1][l] in unalignedWordIndicesInTheLongerName:
                                    unalignedWordIndicesInTheLongerName.remove(jtem[1][l])
                    for k in range(len(item[1])): # the shorter name
                        for l in range(len(jtem[1])): # the longer name
                            # find if the current term in the longer name has already been aligned (before calling alignNamedEntities()), do not align_sentence it in that case
                            alreadyInserted = False
                            for mtem in existingAlignments:
                                if mtem[1] == jtem[1][l]:
                                    alreadyInserted = True
                                    break
                            if jtem[1][l] not in unalignedWordIndicesInTheLongerName or alreadyInserted:
                                continue
                            if [item[1][k], jtem[1][l]] not in alignments  and target[jtem[1][l]-1][2] not in sourceWords  and item[2][k] not in punctuations and jtem[2][l] not in punctuations:
                                alignments.append([item[1][k], jtem[1][l]])

                # else find if the second is a part of the first
                elif is_sublist(jtem[2], item[2]):
                    unalignedWordIndicesInTheLongerName = []
                    for ktem in item[1]:
                        unalignedWordIndicesInTheLongerName.append(ktem)
                    for k in range(len(jtem[2])):
                        for l in range(len(item[2])):
                            if jtem[2][k] == item[2][l] and [item[1][l], jtem[1][k]] not in alignments:
                                alignments.append([item[1][l], jtem[1][k]])
                                if item[1][l] in unalignedWordIndicesInTheLongerName:
                                    unalignedWordIndicesInTheLongerName.remove(item[1][l])
                    for k in range(len(jtem[1])): # the shorter name
                        for l in range(len(item[1])): # the longer name
                            # find if the current term in the longer name has already been aligned (before calling alignNamedEntities()), do not align_sentence it in that case
                            alreadyInserted = False
                            for mtem in existingAlignments:
                                if mtem[0] == item[1][k]:
                                    alreadyInserted = True
                                    break
                            if item[1][l] not in unalignedWordIndicesInTheLongerName or alreadyInserted:
                                continue
                            if [item[1][l], jtem[1][k]] not in alignments  and source[item[1][k]-1][2] not in targetWords  and item[2][l] not in punctuations and jtem[2][k] not in punctuations:
                                alignments.append([item[1][l], jtem[1][k]])

        return alignments


    def alignWords(self, source, target, sourceParseResult, targetParseResult):
    # source and target:: each is a list of elements of the form:
    # [[character begin offset, character end offset], word index, word, lemma, pos tag]

    # function returns the word alignments from source to target - each alignment returned is of the following form:
    # [
    #  [[source word character begin offset, source word character end offset], source word index, source word, source word lemma],
    #  [[target word character begin offset, target word character end offset], target word index, target word, target word lemma]
    # ]

        global punctuations

        sourceWordIndices = [i+1 for i in range(len(source))]
        targetWordIndices = [i+1 for i in range(len(target))]


        alignments = []
        sourceWordIndicesAlreadyAligned = []
        targetWordIndicesAlreadyAligned = []

        sourceWords = [item[2] for item in source]
        targetWords = [item[2] for item in target]

        sourceLemmas = [item[3] for item in source]
        targetLemmas = [item[3] for item in target]

        sourcePosTags = [item[4] for item in source]
        targetPosTags = [item[4] for item in target]


        # align_sentence the sentence ending punctuation first
        if (sourceWords[len(source)-1] in ['.', '!'] and targetWords[len(target)-1] in ['.', '!']) or sourceWords[len(source)-1] == targetWords[len(target)-1]:
            alignments.append([len(source), len(target)])
            sourceWordIndicesAlreadyAligned.append(len(source))
            targetWordIndicesAlreadyAligned.append(len(target))
        elif sourceWords[len(source)-2] in ['.', '!'] and targetWords[len(target)-1] in ['.', '!']:
            alignments.append([len(source)-1, len(target)])
            sourceWordIndicesAlreadyAligned.append(len(source)-1)
            targetWordIndicesAlreadyAligned.append(len(target))
        elif sourceWords[len(source)-1] in ['.', '!'] and targetWords[len(target)-2] in ['.', '!']:
            alignments.append([len(source), len(target)-1])
            sourceWordIndicesAlreadyAligned.append(len(source))
            targetWordIndicesAlreadyAligned.append(len(target)-1)
        elif sourceWords[len(source)-2] in ['.', '!'] and targetWords[len(target)-2] in ['.', '!']:
            alignments.append([len(source)-1, len(target)-1])
            sourceWordIndicesAlreadyAligned.append(len(source)-1)
            targetWordIndicesAlreadyAligned.append(len(target)-1)

        # align_sentence all (>=2)-gram matches with at least one content word
        commonContiguousSublists = find_all_common_contiguous_sublists(sourceWords, targetWords, True)

        for item in commonContiguousSublists:
            allStopWords = True
            for jtem in item:
                if jtem not in cobalt_stopwords and jtem not in punctuations:
                    allStopWords = False
                    break
            if len(item[0]) >= 2 and not allStopWords:
                for j in range(len(item[0])):
                    if item[0][j]+1 not in sourceWordIndicesAlreadyAligned and item[1][j]+1 not in targetWordIndicesAlreadyAligned and [item[0][j]+1, item[1][j]+1] not in alignments:
                        alignments.append([item[0][j]+1, item[1][j]+1])
                        sourceWordIndicesAlreadyAligned.append(item[0][j]+1)
                        targetWordIndicesAlreadyAligned.append(item[1][j]+1)

        # align_sentence hyphenated word groups
        for i in sourceWordIndices:
            if i in sourceWordIndicesAlreadyAligned:
                continue
            if '-' in sourceWords[i-1] and sourceWords[i-1] != '-':
                tokens = sourceWords[i-1].split('-')
                commonContiguousSublists = find_all_common_contiguous_sublists(tokens, targetWords)
                for item in commonContiguousSublists:
                    if len(item[0]) > 1:
                        for jtem in item[1]:
                            if [i, jtem+1] not in alignments:
                                alignments.append([i, jtem+1])
                                sourceWordIndicesAlreadyAligned.append(i)
                                targetWordIndicesAlreadyAligned.append(jtem+1)

        for i in targetWordIndices:
            if i in targetWordIndicesAlreadyAligned:
                continue
            if '-' in target[i-1][2] and target[i-1][2] != '-':
                tokens = target[i-1][2].split('-')
                commonContiguousSublists = find_all_common_contiguous_sublists(sourceWords, tokens)
                for item in commonContiguousSublists:
                    if len(item[0]) > 1:
                        for jtem in item[0]:
                            if [jtem+1, i] not in alignments:
                                alignments.append([jtem+1, i])
                                sourceWordIndicesAlreadyAligned.append(jtem+1)
                                targetWordIndicesAlreadyAligned.append(i)

        # align_sentence named entities
        neAlignments = self.alignNamedEntities(source, target, sourceParseResult, targetParseResult, alignments)
        for item in neAlignments:
            if item not in alignments:
                alignments.append(item)
                if item[0] not in sourceWordIndicesAlreadyAligned:
                    sourceWordIndicesAlreadyAligned.append(item[0])
                if item[1] not in targetWordIndicesAlreadyAligned:
                    targetWordIndicesAlreadyAligned.append(item[1])

        # align_sentence words based on word and dependency match
        sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
        targetDParse = dependencyParseAndPutOffsets(targetParseResult)

        mainVerbAlignments = self.alignPos('verb', 'v', source, target, sourceParseResult, targetParseResult, alignments)
        for item in mainVerbAlignments:
            if item not in alignments:
                alignments.append(item)
                if item[0] not in sourceWordIndicesAlreadyAligned:
                    sourceWordIndicesAlreadyAligned.append(item[0])
                if item[1] not in targetWordIndicesAlreadyAligned:
                    targetWordIndicesAlreadyAligned.append(item[1])

        nounAlignments = self.alignPos('noun', 'n', source, target, sourceParseResult, targetParseResult, alignments)
        for item in nounAlignments:
            if item not in alignments:
                alignments.append(item)
                if item[0] not in sourceWordIndicesAlreadyAligned:
                    sourceWordIndicesAlreadyAligned.append(item[0])
                if item[1] not in targetWordIndicesAlreadyAligned:
                    targetWordIndicesAlreadyAligned.append(item[1])

        adjectiveAlignments = self.alignPos('adjective', 'j', source,  target, sourceParseResult, targetParseResult, alignments)
        for item in adjectiveAlignments:
            if item not in alignments:
                alignments.append(item)
                if item[0] not in sourceWordIndicesAlreadyAligned:
                    sourceWordIndicesAlreadyAligned.append(item[0])
                if item[1] not in targetWordIndicesAlreadyAligned:
                    targetWordIndicesAlreadyAligned.append(item[1])

        adverbAlignments = self.alignPos('adverb', 'r', source, target, sourceParseResult, targetParseResult, alignments)
        for item in adverbAlignments:
            if item not in alignments:
                alignments.append(item)
                if item[0] not in sourceWordIndicesAlreadyAligned:
                    sourceWordIndicesAlreadyAligned.append(item[0])
                if item[1] not in targetWordIndicesAlreadyAligned:
                    targetWordIndicesAlreadyAligned.append(item[1])

        # collect evidence from textual neighborhood for aligning content words
        wordSimilarities = {}
        textualNeighborhoodSimilarities = {}
        sourceWordIndicesBeingConsidered = []
        targetWordIndicesBeingConsidered = []

        for i in sourceWordIndices:
            if i in sourceWordIndicesAlreadyAligned or sourceLemmas[i-1].lower() in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']:
                continue

            for j in targetWordIndices:
                if j in targetWordIndicesAlreadyAligned or targetLemmas[j-1].lower() in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']:
                    continue

                word1 = Word(i, sourceWords[i-1])
                word1.lemma = sourceLemmas[i-1]
                word1.pos = sourcePosTags[i-1]
                word2 = Word(j, targetWords[j-1])
                word2.lemma = targetLemmas[j - 1]
                word2.pos = targetPosTags[j - 1]

                wordSimilarities[(i, j)] = word_relatedness_alignment(word1, word2, self.config)
                sourceWordIndicesBeingConsidered.append(i)
                targetWordIndicesBeingConsidered.append(j)


                # textual neighborhood similarities
                sourceNeighborhood = find_textual_neighborhood(source, i, 3, 3)
                targetNeighborhood = find_textual_neighborhood(target, j, 3, 3)
                evidence = 0

                for k in range(len(sourceNeighborhood[0])):
                    for l in range(len(targetNeighborhood[0])):
                        neighbor1 = Word(sourceNeighborhood[0][k], sourceNeighborhood[1][k])
                        neighbor1.lemma = sourceLemmas[sourceNeighborhood[0][k]-1]
                        neighbor1.pos = sourcePosTags[sourceNeighborhood[0][k]-1]
                        neighbor2 = Word(targetNeighborhood[0][l], targetNeighborhood[1][l])
                        neighbor2.lemma = targetLemmas[targetNeighborhood[0][l]-1]
                        neighbor2.pos = targetPosTags[targetNeighborhood[0][l]-1]
                        if (sourceNeighborhood[1][k] not in cobalt_stopwords + punctuations) and ((sourceNeighborhood[0][k], targetNeighborhood[0][l]) in alignments or (word_relatedness_alignment(neighbor1, neighbor2, self.config) >= self.config.alignment_similarity_threshold)):
                            evidence += word_relatedness_alignment(neighbor1, neighbor2, self.config)
                textualNeighborhoodSimilarities[(i, j)] = evidence

        numOfUnalignedWordsInSource = len(set(sourceWordIndicesBeingConsidered))

        # now align_sentence: find the best alignment in each iteration of the following loop and include in alignments if good enough
        for item in range(numOfUnalignedWordsInSource):
            highestWeightedSim = 0
            bestWordSim = 0
            bestSourceIndex = -1
            bestTargetIndex = -1

            for i in set(sourceWordIndicesBeingConsidered):
                if i in sourceWordIndicesAlreadyAligned:
                    continue

                for j in set(targetWordIndicesBeingConsidered):
                    if j in targetWordIndicesAlreadyAligned:
                        continue

                    if (i, j) not in wordSimilarities:
                        continue

                    if wordSimilarities[(i, j)] == self.config.alignment_similarity_threshold:
                        if wordSimilarities[(i, j)] + textualNeighborhoodSimilarities[(i, j)] <= 1.0:
                            continue

                    if self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * textualNeighborhoodSimilarities[(i, j)] > highestWeightedSim:
                        highestWeightedSim = self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * textualNeighborhoodSimilarities[(i, j)]
                        bestSourceIndex = i
                        bestTargetIndex = j
                        bestWordSim = wordSimilarities[(i, j)]
                        bestTextNeighborhoodSim = textualNeighborhoodSimilarities[(i, j)]

            if bestWordSim >= self.config.alignment_similarity_threshold and [bestSourceIndex, bestTargetIndex] not in alignments and bestSourceIndex not in sourceWordIndicesAlreadyAligned and bestTargetIndex not in targetWordIndicesAlreadyAligned:
                if sourceLemmas[bestSourceIndex-1].lower() not in cobalt_stopwords:
                    alignments.append([bestSourceIndex, bestTargetIndex])
                    sourceWordIndicesAlreadyAligned.append(bestSourceIndex)
                    targetWordIndicesAlreadyAligned.append(bestTargetIndex)

            if bestSourceIndex in sourceWordIndicesBeingConsidered:
                sourceWordIndicesBeingConsidered.remove(bestSourceIndex)
            if bestTargetIndex in targetWordIndicesBeingConsidered:
                targetWordIndicesBeingConsidered.remove(bestTargetIndex)

         # look if any remaining word is a part of a hyphenated word
        for i in sourceWordIndices:
            if i in sourceWordIndicesAlreadyAligned:
                continue
            if '-' in sourceWords[i-1] and sourceWords[i-1] != '-':
                tokens = sourceWords[i-1].split('-')
                commonContiguousSublists = find_all_common_contiguous_sublists(tokens, targetWords)
                for item in commonContiguousSublists:
                    if len(item[0]) == 1 and target[item[1][0]][3] not in cobalt_stopwords:
                        for jtem in item[1]:
                            if [i, jtem+1] not in alignments and jtem+1 not in targetWordIndicesAlreadyAligned:
                                alignments.append([i, jtem+1])
                                sourceWordIndicesAlreadyAligned.append(i)
                                targetWordIndicesAlreadyAligned.append(jtem+1)

        for i in targetWordIndices:
            if i in targetWordIndicesAlreadyAligned:
                continue
            if '-' in target[i-1][2] and target[i-1][2] != '-':
                tokens = target[i-1][2].split('-')
                commonContiguousSublists = find_all_common_contiguous_sublists(sourceWords, tokens)
                for item in commonContiguousSublists:
                    if len(item[0]) == 1 and source[item[0][0]][3] not in cobalt_stopwords:
                        for jtem in item[0]:
                            if [jtem+1, i] not in alignments and i not in targetWordIndicesAlreadyAligned:
                                alignments.append([jtem+1, i])
                                sourceWordIndicesAlreadyAligned.append(jtem+1)
                                targetWordIndicesAlreadyAligned.append(i)

        # collect evidence from dependency neighborhood for aligning stopwords
        wordSimilarities = {}
        dependencyNeighborhoodSimilarities = {}
        sourceWordIndicesBeingConsidered = []
        targetWordIndicesBeingConsidered = []

        for i in sourceWordIndices:
            if sourceLemmas[i-1].lower() not in cobalt_stopwords or i in sourceWordIndicesAlreadyAligned:
                continue

            for j in targetWordIndices:
                if targetLemmas[j-1].lower() not in cobalt_stopwords or j in targetWordIndicesAlreadyAligned:
                    continue

                word1 = Word(i, sourceWords[i-1])
                word1.lemma = sourceLemmas[i - 1]
                word1.pos = sourcePosTags[i-1]
                word2 = Word(j, targetWords[j-1])
                word2.lemma = targetLemmas[j-1]
                word2.pos = targetPosTags[j-1]

                if (sourceLemmas[i-1] != targetLemmas[j-1]) and (word_relatedness_alignment(word1, word2, self.config) < self.config.alignment_similarity_threshold):
                    continue

                wordSimilarities[(i, j)] = word_relatedness_alignment(word1, word2, self.config)

                sourceWordIndicesBeingConsidered.append(i)
                targetWordIndicesBeingConsidered.append(j)

                sourceWordParents = findParents(sourceDParse, i, sourceWords[i-1])
                sourceWordChildren = findChildren(sourceDParse, i, sourceWords[i-1])
                targetWordParents = findParents(targetDParse, j, targetWords[j-1])
                targetWordChildren = findChildren(targetDParse, j, targetWords[j-1])

                evidence = 0

                for item in sourceWordParents:
                    for jtem in targetWordParents:
                        if [item[0], jtem[0]] in alignments:
                            evidence += 1
                for item in sourceWordChildren:
                    for jtem in targetWordChildren:
                        if [item[0], jtem[0]] in alignments:
                            evidence += 1

                dependencyNeighborhoodSimilarities[(i, j)] = evidence

        numOfUnalignedWordsInSource = len(set(sourceWordIndicesBeingConsidered))

        # now align_sentence: find the best alignment in each iteration of the following loop and include in alignments if good enough
        for item in range(numOfUnalignedWordsInSource):
            highestWeightedSim = 0
            bestWordSim = 0
            bestSourceIndex = -1
            bestTargetIndex = -1

            for i in set(sourceWordIndicesBeingConsidered):
                for j in set(targetWordIndicesBeingConsidered):
                    if (i, j) not in wordSimilarities:
                        continue

                    if self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * dependencyNeighborhoodSimilarities[(i, j)] > highestWeightedSim:
                        highestWeightedSim = self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * dependencyNeighborhoodSimilarities[(i, j)]
                        bestSourceIndex = i
                        bestTargetIndex = j
                        bestWordSim = wordSimilarities[(i, j)]
                        bestDependencyNeighborhoodSim = dependencyNeighborhoodSimilarities[(i, j)]

            if bestWordSim >= self.config.alignment_similarity_threshold and bestDependencyNeighborhoodSim > 0 and [bestSourceIndex, bestTargetIndex] not in alignments and bestSourceIndex not in sourceWordIndicesAlreadyAligned and bestTargetIndex not in targetWordIndicesAlreadyAligned:
                alignments.append([bestSourceIndex, bestTargetIndex])
                sourceWordIndicesAlreadyAligned.append(bestSourceIndex)
                targetWordIndicesAlreadyAligned.append(bestTargetIndex)

            if bestSourceIndex in sourceWordIndicesBeingConsidered:
                sourceWordIndicesBeingConsidered.remove(bestSourceIndex)
            if bestTargetIndex in targetWordIndicesBeingConsidered:
                targetWordIndicesBeingConsidered.remove(bestTargetIndex)

        # collect evidence from textual neighborhood for aligning stopwords and punctuations
        wordSimilarities = {}
        textualNeighborhoodSimilarities = {}
        sourceWordIndicesBeingConsidered = []
        targetWordIndicesBeingConsidered = []

        for i in sourceWordIndices:
            if (sourceLemmas[i-1].lower() not in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']) or i in sourceWordIndicesAlreadyAligned:
                continue

            for j in targetWordIndices:
                if (targetLemmas[j-1].lower() not in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']) or j in targetWordIndicesAlreadyAligned:
                    continue

                word1 = Word(i, sourceWords[i-1])
                word1.lemma = sourceLemmas[i-1]
                word1.pos = sourcePosTags[i-1]
                word2 = Word(j, targetWords[j-1])
                word2.lemma = targetLemmas[j-1]
                word2.pos = targetPosTags[j-1]

                if word_relatedness_alignment(word1, word2, self.config) < self.config.alignment_similarity_threshold:
                    continue


                wordSimilarities[(i, j)] = word_relatedness_alignment(word1, word2, self.config)

                sourceWordIndicesBeingConsidered.append(i)
                targetWordIndicesBeingConsidered.append(j)

                # textual neighborhood evidence, increasing evidence if content words around this stop word are aligned
                evidence = 0

                k = i
                l = j

                while k > 0:
                    if sourceLemmas[k-1].lower() in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']:
                        k -= 1
                    else:
                        break
                while l > 0:
                    if targetLemmas[l-1].lower() in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']:
                        l -= 1
                    else:
                        break

                m = i
                n = j


                while m < len(sourceLemmas) - 1:
                    if sourceLemmas[m - 1].lower() in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']:

                        m += 1
                    else:
                        break
                while n < len(targetLemmas) - 1:
                    if targetLemmas[n - 1].lower() in cobalt_stopwords + punctuations + ['\'s', '\'d', '\'ll']:
                        n += 1
                    else:
                        break

                if [k, l] in alignments:
                    evidence += 1

                if [m, n] in alignments:
                    evidence += 1

                textualNeighborhoodSimilarities[(i, j)] = evidence

        numOfUnalignedWordsInSource = len(set(sourceWordIndicesBeingConsidered))

        # now align_sentence: find the best alignment in each iteration of the following loop and include in alignments if good enough
        for item in range(numOfUnalignedWordsInSource):
            highestWeightedSim = 0
            bestWordSim = 0
            bestSourceIndex = -1
            bestTargetIndex = -1

            for i in set(sourceWordIndicesBeingConsidered):
                if i in sourceWordIndicesAlreadyAligned:
                    continue

                for j in set(targetWordIndicesBeingConsidered):
                    if j in targetWordIndicesAlreadyAligned:
                        continue

                    if (i, j) not in wordSimilarities:
                        continue

                    if self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * textualNeighborhoodSimilarities[(i, j)] > highestWeightedSim:
                        highestWeightedSim = self.config.theta * wordSimilarities[(i, j)] + (1 - self.config.theta) * textualNeighborhoodSimilarities[(i, j)]
                        bestSourceIndex = i
                        bestTargetIndex = j
                        bestWordSim = wordSimilarities[(i, j)]
                        bestTextNeighborhoodSim = textualNeighborhoodSimilarities[(i, j)]

            if bestWordSim >= self.config.alignment_similarity_threshold and bestTextNeighborhoodSim > 0 and [bestSourceIndex, bestTargetIndex] not in alignments and bestSourceIndex not in sourceWordIndicesAlreadyAligned and bestTargetIndex not in targetWordIndicesAlreadyAligned:
                alignments.append([bestSourceIndex, bestTargetIndex])
                sourceWordIndicesAlreadyAligned.append(bestSourceIndex)
                targetWordIndicesAlreadyAligned.append(bestTargetIndex)

            if bestSourceIndex in sourceWordIndicesBeingConsidered:
                sourceWordIndicesBeingConsidered.remove(bestSourceIndex)
            if bestTargetIndex in targetWordIndicesBeingConsidered:
                targetWordIndicesBeingConsidered.remove(bestTargetIndex)

        alignments = [item for item in alignments if item[0] != 0 and item[1] != 0]

        return alignments

    def compileContextInfo(self, sourceWord, sourceIndex, targetWord, targetIndex, sourceSentence, targetSentence, sourceParseResult, targetParseResult, alignments):
        sourceLemmas = [item[3] for item in sourceSentence]
        targetLemmas = [item[3] for item in targetSentence]

        sourcePosTags = [item[4] for item in sourceSentence]
        targetPosTags = [item[4] for item in targetSentence]

        sourceDParse = dependencyParseAndPutOffsets(sourceParseResult)
        targetDParse = dependencyParseAndPutOffsets(targetParseResult)

        totalContextSourceLabels = []
        totalContextTargetLabels = []

        contextInfo = {}

        pos = ''
        if sourceWord[4].lower().startswith('v'):
            pos = 'verb'
        if sourceWord[4].lower().startswith('n'):
            pos = 'noun'
        if sourceWord[4].lower().startswith('j'):
            pos = 'adjective'
        if sourceWord[4].lower().startswith('r'):
            pos = 'adverb'

        sourceWordParents = findParents(sourceDParse, sourceIndex, sourceWord[2])
        sourceWordChildren = findChildren(sourceDParse, sourceIndex, sourceWord[2])
        targetWordParents = findParents(targetDParse, targetIndex, targetWord[2])
        targetWordChildren = findChildren(targetDParse, targetIndex, targetWord[2])

        for item in sourceWordChildren + sourceWordParents:
            totalContextSourceLabels.append(item[2])

        for item in targetWordChildren + targetWordParents:
            totalContextTargetLabels.append(item[2])

        contextInfo['srcDiff'] = self.findDependencyDifference(pos, sourceWord[2], sourceIndex, targetWord[2], targetIndex, sourceDParse, targetDParse, alignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)[0]
        contextInfo['tgtDiff'] = self.findDependencyDifference(pos, sourceWord[2], sourceIndex, targetWord[2], targetIndex, sourceDParse, targetDParse, alignments, sourcePosTags, targetPosTags, sourceLemmas, targetLemmas)[1]
        contextInfo['srcCon'] = totalContextSourceLabels
        contextInfo['tgtCon'] = totalContextTargetLabels

        return contextInfo

    def align_sentence(self, sentence1, sentence2):

        sentence1ParseResult = parse_text(sentence1)
        sentence2ParseResult = parse_text(sentence2)

        sentence1LemmasAndPosTags = prepareSentence(sentence1)
        sentence2LemmasAndPosTags = prepareSentence(sentence2)

        myWordAlignments = sorted(self.alignWords(sentence1LemmasAndPosTags, sentence2LemmasAndPosTags, sentence1ParseResult, sentence2ParseResult), key=itemgetter(0))
        myWordAlignmentTokens = [[sentence1LemmasAndPosTags[item[0]-1][2], sentence2LemmasAndPosTags[item[1]-1][2]] for item in myWordAlignments]

        # To test external context compiler
        # external_compiler = ContextInfoCompiler('english')
        # source_words = StanfordParseLoader._process_parse_result(sentence1ParseResult['sentences'][0])
        # target_words = StanfordParseLoader._process_parse_result(sentence2ParseResult['sentences'][0])
        # contextInfo = external_compiler.compile_context_info(source_words, target_words, myWordAlignments)

        contextInfo = []
        for pair in myWordAlignments:
            sourceWord = sentence1LemmasAndPosTags[pair[0] - 1]
            targetWord = sentence2LemmasAndPosTags[pair[1] - 1]
            contextInfo.append(self.compileContextInfo(sourceWord,  pair[0], targetWord, pair[1], sentence1LemmasAndPosTags, sentence2LemmasAndPosTags, sentence1ParseResult, sentence2ParseResult, myWordAlignments))

        return [myWordAlignments, myWordAlignmentTokens, contextInfo]

    def align_documents(self, tgt, ref):

        tst_phrases = read_parsed_sentences(codecs.open(os.path.expanduser(tgt), 'r', encoding='UTF-8'))
        ref_phrases = read_parsed_sentences(codecs.open(os.path.expanduser(ref), 'r', encoding='UTF-8'))

        if "paraphrases" in self.config.selected_lexical_resources:
            load_resources.load_ppdb(self.config.path_to_ppdb)
        if "distributional" in self.config.selected_lexical_resources:
            load_resources.load_word_vectors(self.config.path_to_vectors)

        for i, candidate in enumerate(tst_phrases):
            self.alignments.append(self.align_sentence(candidate, ref_phrases[i]))
            print(str(i))

    def write_alignments(self, output_file_name):

        my_output = codecs.open(os.path.expanduser(output_file_name), 'w', 'utf-8')

        for i, sentence in enumerate(self.alignments):
            print('Sentence #' + str(i + 1), file=my_output)

            for j, widx in enumerate(self.alignments[i][0]):

                my_output.write(str(widx) + ' : ' + '[' + self.alignments[i][1][j][0] + ', ' + self.alignments[i][1][j][1] + ']' + ' : ')
                my_output.write('srcDiff=' + ','.join(self.alignments[i][2][j]['srcDiff']) + ';')
                my_output.write('srcCon=' + ','.join(self.alignments[i][2][j]['srcCon']) + ';')
                my_output.write('tgtDiff=' + ','.join(self.alignments[i][2][j]['tgtDiff']) + ';')
                my_output.write('tgtCon=' + ','.join(self.alignments[i][2][j]['tgtCon']) + '\n')

        my_output.close()
