from utils.sentence import Sentence
from lex_resources.config import *


class CleanPunctuation(object):


    def get_previous_punctuations(self, idx, tokens):

        count = 0
        for token in tokens[:idx]:
            if token in punctuations:
                count += 1

        return count


    def get_clean_alignments(self, punct_cand, punct_ref, cand, ref):

        alignments = [[], [], []]

        for d_type in range(len(cand['alignments'])):

            for i, w_pair in enumerate(cand['alignments'][d_type]):

                if cand['alignments'][0][i][0] - 1 in punct_cand or cand['alignments'][0][i][1] - 1 in punct_ref:
                    continue

                if d_type == 0:

                    cand_punct = self.get_previous_punctuations(w_pair[0] - 1, cand['tokens'])
                    ref_punct = self.get_previous_punctuations(w_pair[0] - 1, ref['tokens'])
                    alignments[d_type].append([w_pair[0] - cand_punct, w_pair[1] - ref_punct])
                else:
                    alignments[d_type].append(w_pair)

        return alignments

    def get_clean_data(self, data, punct_idxs):

        clean = []
        for i, wobject in enumerate(data):
            if i in punct_idxs:
                continue
            clean.append(wobject)

        return clean

    def get_clean_sentence(self, punct_cand, punct_ref, cand, ref):

        clean_cand = Sentence()
        clean_ref = Sentence()

        for method in sorted(cand.keys()):
            if method == 'alignments':
                alignments = self.get_clean_alignments(punct_cand, punct_ref, cand, ref)
                clean_data_cand = alignments
                clean_data_ref = alignments
            else:
                clean_data_cand = self.get_clean_data(cand[method], punct_cand)
                clean_data_ref = self.get_clean_data(ref[method], punct_ref)

            clean_cand.add_data(method, clean_data_cand)
            clean_ref.add_data(method, clean_data_ref)

        return clean_cand, clean_ref

    @staticmethod
    def get_punctuations(tokens):

            punct_idxs = []
            for i, token in enumerate(tokens):
                if token in punctuations:
                    punct_idxs.append(i)

            return punct_idxs

    def clean_punctuation(self, cand, ref):

        punct_cand = self.get_punctuations(cand['tokens'])
        punct_ref = self.get_punctuations(ref['tokens'])

        clean_cand, clean_ref = self.get_clean_sentence(punct_cand, punct_ref, cand, ref)

        return [clean_cand, clean_ref]




