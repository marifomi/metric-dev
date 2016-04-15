from __future__ import division

# -*- coding: utf-8 -*-
# Natural Language Toolkit: BLEU
#
# Copyright (C) 2001-2013 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

import math
import os
import numpy as np
from nltk import word_tokenize
from nltk.compat import Counter
from nltk.util import ngrams


class BLEU(object):

    @staticmethod
    def compute(candidate, references):
        candidate = [c.lower() for c in candidate]
        references = [[r.lower() for r in reference] for reference in references]
        ps = []
        for i in range(1, 5):
            ps.append(BLEU.modified_precision(candidate, references, i))

        smoothed_ps = BLEU.smooth(ps)
        p = np.prod(smoothed_ps) ** 0.25

        bp = BLEU.brevity_penalty(candidate, references)
        return p * bp

    @staticmethod
    def modified_precision(candidate, references, n):

        counts = Counter(ngrams(candidate, n))

        if not counts:
            return 0, 0

        max_counts = {}
        for reference in references:
            reference_counts = Counter(ngrams(reference, n))
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

        clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

        return sum(clipped_counts.values()), sum(counts.values())

    @staticmethod
    def smooth(ps):

        k = 1
        smoothed = []
        for match, total in ps:
            if total == 0:
                smoothed.append(1.0)
            elif match == 0:
                smoothed.append((1 / 2 ** k) / float(total))
                k += 1
            else:
                smoothed.append(match / float(total))

        return smoothed

    @staticmethod
    def brevity_penalty(candidate, references):
        c = len(candidate)
        r = min(len(r) for r in references)

        if c > r:
            return 1
        else:
            return math.exp(1 - r / c)

if __name__ == "__main__":

    # candidate = open(os.getcwd() + '/test/tgt.token').readlines()
    # reference = open(os.getcwd() + '/test/ref.token').readlines()
    # output = open('bleu_test,txt', 'w')
    #
    # for i, sentence in enumerate(candidate):
    #
    #     bleu = BLEU.compute(sentence.strip().split(' '), [reference[i].strip().split(' ')])
    #     output.write(str(bleu) + '\n')

    candidate1 = ["it", "'s", "not", ",", "mr", ".", "hašku", "."]
    reference1 = ["this", "is", "not", "the", "way", ",", "mr", "hašek", "."]
    print(str(BLEU.compute(candidate1, [reference1])))
