__author__ = 'MarinaFomicheva'

import subprocess
import codecs
import os
from collections import OrderedDict
from collections import defaultdict
from math import floor
import re

""" Produce n-gram counts from corpus
    Clean n-gram counts from corpus
    Produce lm from corpus """


class LanguageModel(object):

    def __init__(self):
        self.ngram_tools_path = ''
        self.ngrams = OrderedDict()
        self.freqs = defaultdict(list)
        self.cut_offs = defaultdict(dict)

    def set_path_to_tools(self, path):
        self.ngram_tools_path = path

    def produce_lm(self, corpus_file_name, ngram_size):

        if not os.path.exists(corpus_file_name):
            print('File does not exist!')

        SRILM = [self.ngram_tools_path + '/' + 'ngram-count', '-order', str(ngram_size), '-text', corpus_file_name, '-lm', corpus_file_name + '.lm']
        # -interpolate ? -kndiscount ?
        subprocess.check_call(SRILM)

    def produce_ngram_counts(self, corpus_file_name, ngram_size):

        SRILM = [self.ngram_tools_path + '/' + 'ngram-count', '-order', str(ngram_size), '-text', corpus_file_name, '-write', corpus_file_name + '.ngram']
        subprocess.check_call(SRILM)

    def produce_ppl(self, input_file, output_file, lm_file, ngram_size, debug=2):

        if os.path.exists(output_file):
            print('File with lm perplexities already exist')
            return

        my_output = open(output_file, 'w')

        SRILM = [self.ngram_tools_path + '/' + 'ngram', '-lm', lm_file, '-order', str(ngram_size), '-debug', str(debug), '-ppl', input_file]
        subprocess.check_call(SRILM, stdout=my_output)

    def sort_ngram_counts(self, min_freq, raw_count_file_name):

        lines = open(raw_count_file_name, 'r')
        for line in lines:
            my_freq = int(line.strip().split('\t')[1])
            my_ngram_size = len(line.strip().split('\t')[0].split(' ')) - 1

            if my_freq > min_freq:
                self.ngrams[line.strip().split('\t')[0]] = my_freq
                self.freqs[my_ngram_size].append(my_freq)

        lines.close()

    def compute_cut_offs(self, slice_number, ngram_size):

        for i in range(ngram_size):
            self.freqs[i].sort(key=int)
            my_size = len(self.freqs[i]) - 1

            for j in range(slice_number):
                my_cut_off = int(floor((j + 1) * my_size / slice_number))
                self.cut_offs[i][j] = self.freqs[i][my_cut_off]

    def write_clean_ngram_counts(self, raw_count_file_name, slice_number, ngram_size):

        my_output = open(raw_count_file_name + '.clean', 'w')

        for i in range(ngram_size):
            my_output.write(str(i) + '-gram\t')

            for val in range(slice_number):
                my_output.write(str(self.cut_offs[i][val]) + '\t')

            my_output.write('\n')

        for key in self.ngrams.keys():
            my_output.write(key + '\t' + str(self.ngrams[key]) + '\n')

        my_output.close()

    def tokenize(self, tokenizer, input_file_name):

        myinput = open(input_file_name, 'r')
        myoutput = open(input_file_name + '.tok', 'w')
        p = subprocess.Popen(['perl', tokenizer], stdin=myinput, stdout=myoutput)
        p.wait()
        myoutput.flush()

        print("Tokenization finished")

    def read_ppl(self, ppl_file_name):

        lm_features = defaultdict(list)
        ppl_file = open(ppl_file_name, 'r')

        line_counter = 0

        for line in ppl_file:
            if re.sub('.ppl', '', ppl_file_name) in line:
                break
            if 'OOVs' in line:
                lm_features[line_counter].append(int(line.strip().split(' ')[4]))
                line_counter += 1
            if 'logprob' in line:
                lm_features[line_counter] += [float(line.strip().split(' ')[3]), float(line.strip().split(' ')[5]), float(line.strip().split(' ')[7])]
                print(line.strip())

        return lm_features


def main():

    f_in = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'system.pos.join.train'
    f_corpus = os.path.expanduser('~/Dropbox/workspace/questplusplus/lang_resources/english/arabic_english_data/TrainingFileFiltred.sgm.En.Tok.LC.cut1-100')
    lm = LanguageModel()
    lm.produce_ngram_counts(f_corpus, 3)
    lm.sort_ngram_counts(2, f_corpus + "." + "ngram")
    lm.compute_cut_offs(4, 3)
    lm.write_clean_ngram_counts(f_corpus + "." + "ngram", 4, 3)


if __name__ == '__main__':
    main()
