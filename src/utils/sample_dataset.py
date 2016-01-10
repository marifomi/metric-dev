__author__ = 'MarinaFomicheva'

from sklearn import cross_validation as cv
from src.learning import learn_model
import os
import numpy as np

def write_sampled_phrs(size, file1, file2):

    sampled_phrs = cv.train_test_split(range(size))
    out_train = open(file1, 'w')
    out_test = open(file2, 'w')

    for phr in sorted(sampled_phrs[0]):
        out_train.write(str(phr) + '\n')
    for phr in sorted(sampled_phrs[1]):
        out_test.write(str(phr) + '\n')

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

if __name__ == '__main__':
    # sample(1120)

    predicted = learn_model.run(os.getcwd() + '/' + 'config' + '/' + 'learner' + '/' + 'svr.cfg',
                                x_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/output/features_word_level.tsv.0',
                                x_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/output/features_word_level.tsv.1',
                                y_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/output/sample.seg.ad.stnd.all-en.tsv.0',
                                y_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/output/sample.seg.ad.stnd.all-en.tsv.1'
                                )

    # create_samples(os.getcwd() + '/' + 'output' + '/' + 'features_word_level.tsv', os.getcwd() + '/' + 'output' + '/' + 'sample.seg.ad.stnd.all-en.tsv',
    #            os.getcwd() + '/' + 'output' + '/' + 'train_phrases.txt',
    #            os.getcwd() + '/' + 'output' + '/' + 'test_phrases.txt'
    #            )
