__author__ = 'MarinaFomicheva'

from sklearn import cross_validation as cv
from src.learning import learn_model
import os
import numpy as np
from src.utils.core_nlp_utils import read_parsed_sentences
import codecs

def save_sampled_phrs(sampled_phrs, file1, file2):

    if os.path.exists(file1) or os.path.exists(file2):
        print "Samples already exist"
        return

    out_train = open(file1, 'w')
    out_test = open(file2, 'w')

    for phr in sorted(sampled_phrs[0]):
        out_train.write(str(phr) + '\n')
    for phr in sorted(sampled_phrs[1]):
        out_test.write(str(phr) + '\n')

    out_train.close()
    out_test.close()

def print_sampled_data(f_in, f_train_phrs, f_test_phrs, format=None):

    phrs_train = [int(x) for x in open(f_train_phrs).readlines()]
    phrs_test = [int(x) for x in open(f_test_phrs).readlines()]
    out_train = codecs.open(f_in + '.train', 'w', 'utf-8')
    out_test = codecs.open(f_in + '.test', 'w', 'utf-8')

    if format == 'parsed':
        sentences = read_parsed_sentences(codecs.open(f_in, 'r', 'utf-8'))
    else:
        sentences = codecs.open(f_in, 'r', 'utf-8').readlines()

    for tr_phr in phrs_train:
        out_train.write(sentences[tr_phr])
    for te_phr in phrs_test:
        out_test.write(sentences[te_phr])

    out_train.close()
    out_test.close()


def create_samples(f_features, f_obj, f_train_phrs, f_test_phrs):

    features = open(f_features, 'r').readlines()
    objective = open(f_obj, 'r').readlines()

    train_phrs = [int(x) for x in open(f_train_phrs, 'r').readlines()]
    test_phrs = [int(x) for x in open(f_test_phrs, 'r').readlines()]

    samples = [train_phrs, test_phrs]

    for i in range(2):
        feats_o = open(f_features + '.' + str(i), 'w')
        obj_o = open(f_obj + '.' + str(i), 'w')

        for idx in sorted(samples[i]):
            feats_o.write(features[idx])
            obj_o.write(objective[idx])

        feats_o.close()
        obj_o.close()

def main():

    file1 = os.getcwd() + '/' + 'data' + '/' + 'mtc4' + '/' + 'phr_train.txt'
    file2 = os.getcwd() + '/' + 'data' + '/' + 'mtc4' + '/' + 'phr_test.txt'

    multi_bleu = os.getcwd() + '/' + 'data' + '/' + 'mtc4' + '/' + 'bleu_ref_multi.txt'
    print_sampled_data(multi_bleu, file1, file2)


if __name__ == '__main__':
    main()
