import numpy as np

from features.impl.abstract_feature import *
from utils import word_sim


class CountNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, 'Returns an count of non aligned chunks')

    def run(self, cand, ref):
        count = 0
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_ref):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                count += 1
            prev = i

        if prev is None:
            prev = 0

        if prev < len(ref['tokens']):
            count += 1

        AbstractChunkFeature.set_value(self, count)


class CountNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, 'Returns an count of non aligned chunks on cadidate side')

    def run(self, cand, ref):
        count = 0
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[0])

        prev = None

        for i in sorted(aligned_tokens_cand):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                count += 1
            prev = i

        if prev is None:
            prev = 0

        if prev < len(ref['tokens']):
            count += 1

        AbstractChunkFeature.set_value(self, count)


class AvgLengthNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'avg_length_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, 'Returns an average length of non aligned chunks')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_ref):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                counted.append(i - 1 if prev is None else i - prev - 1)
            prev = i

        if prev is None:
            prev = 0

        if prev < len(ref['tokens']):
            counted.append(len(ref['tokens']) - prev)

        AbstractChunkFeature.set_value(self, np.mean(counted) if len(counted) > 0 else 0)


class AvgLengthNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'avg_length_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, 'Returns an average length of non aligned chunks')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[0])

        prev = None

        for i in sorted(aligned_tokens_cand):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                counted.append(i - 1 if prev is None else i - prev - 1)
            prev = i

        if prev is None:
            prev = 0

        if prev < len(cand['tokens']):
            counted.append(len(cand['tokens']) - prev)

        AbstractChunkFeature.set_value(self, np.mean(counted) if len(counted) > 0 else 0)


class AvgDistanceBetweenNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'avg_distance_between_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, 'Returns an average distance between first X non aligned chunks '
                                                   'in reference translation in words')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None
        for i in sorted(aligned_tokens_ref):
            if prev is not None and prev < i - 1:
                counted.append(i - prev - 1)
            prev = i

        AbstractChunkFeature.set_value(self, np.mean(counted) if len(counted) > 0 else 0)


class AvgDistanceBetweenNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'avg_distance_between_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, 'Returns an average distance between first X non aligned chunks '
                                                   'in candidate translation in words')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[0])

        prev = None
        for i in sorted(aligned_tokens_cand):
            if prev is not None and prev < i - 1:
                counted.append(i - prev - 1)
            prev = i

        AbstractChunkFeature.set_value(self, np.mean(counted) if len(counted) > 0 else 0)


class CountWordsInNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_words_in_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, 'Returns an array of lengths of first X non aligned chunks')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_ref):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                counted.append(i - 1 if prev is None else i - prev - 1)
            if len(counted) == AbstractChunkFeature.chunk_number:
                break
            prev = i

        if prev is None:
            prev = 0

        if prev < len(ref['tokens']) and len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(len(ref['tokens']) - prev)

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class CountWordsInNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_words_in_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, 'Returns an array of lengths of first X non aligned chunks in candidate translation')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[0])

        prev = None

        for i in sorted(aligned_tokens_cand):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                counted.append(i - 1 if prev is None else i - prev - 1)
            if len(counted) == AbstractChunkFeature.chunk_number:
                break
            prev = i

        if prev is None:
            prev = 0

        if prev < len(cand['tokens']) and len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(len(cand['tokens']) - prev)

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class DistanceBetweenNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'distance_between_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, 'Returns an array of distances between first X non aligned chunks '
                                                   'in reference translation in words')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None
        for i in sorted(aligned_tokens_ref):
            if prev is not None and prev < i - 1:
                counted.append(i - prev - 1)
            prev = i

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class DistanceBetweenNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'distance_between_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, 'Returns an array of distances between first X non aligned chunks '
                                                   'in candidate translation in words')

    def run(self, cand, ref):
        counted = []
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[0])

        prev = None
        for i in sorted(aligned_tokens_cand):
            if prev is not None and prev < i - 1:
                counted.append(i - prev - 1)
            prev = i

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class CountFunctionWordsInNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_function_words_in_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, "Number of function words in each non aligned "
                                                   "chunk on reference side")

    def run(self, cand, ref):
        counted = []
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_ref):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                count = 0
                if prev is None:
                    prev = 0
                for j in range(prev, i - prev - 1):
                    if word_sim.function_word(ref['tokens'][j]):
                        count += 1
                counted.append(count)
            if len(counted) == AbstractChunkFeature.chunk_number:
                break
            prev = i

        if prev is None:
            prev = 0

        if prev < len(ref['tokens']) and len(counted) < AbstractChunkFeature.chunk_number:
            count = 0
            for j in range(prev, len(ref['tokens']) - 1):
                if word_sim.function_word(ref['tokens'][j]):
                    count += 1
            counted.append(count)

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class CountFunctionWordsInNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_function_words_in_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, "Number of function words in each non aligned "
                                                   "chunk on candidate side")

    def run(self, cand, ref):
        counted = []
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_cand):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                count = 0
                if prev is None:
                    prev = 0
                for j in range(prev, i - prev - 1):
                    if word_sim.function_word(ref['tokens'][j]):
                        count += 1
                counted.append(count)
            if len(counted) == AbstractChunkFeature.chunk_number:
                break
            prev = i

        if prev is None:
            prev = 0

        if prev < len(cand['tokens']) and len(counted) < AbstractChunkFeature.chunk_number:
            count = 0
            for j in range(prev, len(cand['tokens']) - 1):
                if word_sim.function_word(cand['tokens'][j]):
                    count += 1
            counted.append(count)

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class CountContentWordsInNonAlignedChunksRef(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_content_words_in_non_aligned_chunks_ref')
        AbstractChunkFeature.set_description(self, "Number of content words in each non aligned "
                                                   "chunk on reference side")

    def run(self, cand, ref):
        counted = []
        aligned_tokens_ref = []

        for a in ref['alignments'][0]:
            aligned_tokens_ref.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_ref):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                count = 0
                if prev is None:
                    prev = 0
                for j in range(prev, i - prev - 1):
                    if not word_sim.function_word(ref['tokens'][j]):
                        count += 1
                counted.append(count)
            if len(counted) == AbstractChunkFeature.chunk_number:
                break
            prev = i

        if prev is None:
            prev = 0

        if prev < len(ref['tokens']) and len(counted) < AbstractChunkFeature.chunk_number:
            count = 0
            for j in range(prev, len(ref['tokens']) - 1):
                if not word_sim.function_word(ref['tokens'][j]):
                    count += 1
            counted.append(count)

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)


class CountContentWordsInNonAlignedChunksCand(AbstractChunkFeature):

    def __init__(self):
        AbstractChunkFeature.__init__(self)
        AbstractChunkFeature.set_name(self, 'count_content_words_in_non_aligned_chunks_cand')
        AbstractChunkFeature.set_description(self, "Number of content words in each non aligned "
                                                   "chunk on candidate side")

    def run(self, cand, ref):
        counted = []
        aligned_tokens_cand = []

        for a in cand['alignments'][0]:
            aligned_tokens_cand.append(a[1])

        prev = None

        for i in sorted(aligned_tokens_cand):
            if (prev is None and i > 1) or (prev is not None and (i != prev + 1 and i != prev)):
                count = 0
                if prev is None:
                    prev = 0
                for j in range(prev, i - prev - 1):
                    if not word_sim.function_word(ref['tokens'][j]):
                        count += 1
                counted.append(count)
            if len(counted) == AbstractChunkFeature.chunk_number:
                break
            prev = i

        if prev is None:
            prev = 0

        if prev < len(cand['tokens']) and len(counted) < AbstractChunkFeature.chunk_number:
            count = 0
            for j in range(prev, len(cand['tokens']) - 1):
                if not word_sim.function_word(cand['tokens'][j]):
                    count += 1
            counted.append(count)

        while len(counted) < AbstractChunkFeature.chunk_number:
            counted.append(0)

        AbstractChunkFeature.set_value(self, counted)
