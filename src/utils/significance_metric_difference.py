__author__ = 'MarinaFomicheva'

import scipy.stats as stats
import math
import os


class SignificanceMetricDifference(object):

    @staticmethod
    def read_file(lines):

        scores = []
        for line in lines:
            scores.append(float(line.strip()))

        return scores

    def get_scores(self, f_metric1, f_metric2, f_human):

        result = [[], [], []]

        m1 = open(f_metric1, 'r').readlines()
        m2 = open(f_metric2, 'r').readlines()
        hum = open(f_human, 'r').readlines()

        result[0] = self.read_file(m1)
        result[1] = self.read_file(m2)
        result[2] = self.read_file(hum)

        return result

    def test_significance(self, scores_metric1, scores_metric2, human_scores):

        r12 = self.pearson(scores_metric1, scores_metric2)
        r13 = self.pearson(scores_metric1, human_scores)
        r23 = self.pearson(scores_metric2, human_scores)

        n = len(scores_metric1)
        k = 1 - math.pow(r12, 2) - math.pow(r13, 2) - math.pow(r23, 2) + 2 * r12 * r13 * r23
        t = (r13 - r23) * math.sqrt((n - 1) * (1 + r12)) / math.sqrt((2 * k * (n - 1) / (n - 3)) + (pow((r23 + r13), 2) / 4) * pow((1 - r12), 3))
        pvalue = stats.t.sf(math.fabs(t), n - 1) * 2

        return (r13, r23, pvalue)

    @staticmethod
    def pearson(list1, list2):
        return stats.pearsonr(list1, list2)[0]

def main():

    fm1 = os.getcwd() + '/' + 'results' + '/' + 'mtc4' + '/' + 'test_correlation' + '/' + 'bleu_simple.tsv'
    fm2 = os.getcwd() + '/' + 'results' + '/' + 'mtc4' + '/' + 'test_correlation' + '/' + 'quest_svm_human'
    fhum = os.getcwd() + '/' + 'results' + '/' + 'mtc4' + '/' + 'test_correlation' + '/' + 'human.test'

    signif = SignificanceMetricDifference()
    m1, m2, hum = signif.get_scores(fm1, fm2, fhum)
    r13, r23, pvalue = signif.test_significance(m1, m2, hum)
    print str(r13)
    print str(r23)
    print str(pvalue)


if __name__ == '__main__':
    main()
