__author__ = 'MarinaFomicheva'

from sklearn import cross_validation as cv
from src.learning import learn_model
import os

def train_test(feats, obj):

    feats_ = open(feats, 'r').readlines()
    obj_ = open(obj, 'r').readlines()

    samples = cv.train_test_split(range(len(feats_)))

    for i in range(2):
        sample_o = open('sample' + '.' + str(i), 'w')

        for idx in sorted(samples[i]):
            sample_o.write(str(idx) + '\n')
        sample_o.close()

    # for i in range(2):
    #     feats_o = open(feats + '.' + str(i), 'w')
    #     obj_o = open(obj + '.' + str(i), 'w')
    #
    #     for idx in sorted(samples[i]):
    #         feats_o.write(feats_[idx])
    #         obj_o.write(obj_[idx])
    #
    # feats_o.close()
    # obj_o.close()
    #
    # predicted = learn_model.run(os.getcwd() + '/' + 'config' + '/' + 'learner' + '/' + 'my_svr.cfg')
    # for pred in predicted:
    #     print(str(pred))

if __name__ == '__main__':
    train_test(os.getcwd() + '/' + 'output' + '/' + 'features.tsv', os.getcwd() + '/' + 'output' + '/' + 'sample.seg.ad.stnd.all-en.tsv')
