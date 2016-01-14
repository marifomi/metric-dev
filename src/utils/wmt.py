__author__ = 'MarinaFomicheva'

import os
from src.tools.run_tools import RunTools
from src.features.feature_extractor import FeatureExtractor
from src.tools.human_rank import HumanRank
from src.learning import learn_model
from sklearn.linear_model import LogisticRegression
from src.learning.features_file_utils import read_features_file
from src.learning.features_file_utils import read_reference_file
from src.utils.plain_arff import *
from weka_logistic_wrapper import *

class WMT(object):

    def __init__(self, dir_, dataset, lps):
        self.dir = dir_
        self.dataset = dataset
        self.lps = lps

    def process_wmt(self, f_output_feats, f_output_meta_data):

        features = open(f_output_feats, 'w')
        meta_data = open(f_output_meta_data, 'w')

        for lp in sorted(os.listdir(self.dir + '/' + 'parsed' + '/' + 'system-outputs' + '/' + self.dataset)):
            if lp.startswith('.DS'):
                continue
            if lp not in self.lps:
                continue
            ref_file_parse = set_ref_file(self.dir, 'parsed', self.dataset, lp)
            ref_file_plain = set_ref_file(self.dir, 'plain', self.dataset, lp)

            for sys in sorted(os.listdir(self.dir + '/' + 'parsed' + '/' + 'system-outputs' + '/' + self.dataset + '/' + lp)):
                print sys
                sys_file_parse = self.dir + '/' + 'parsed' + '/' + 'system-outputs' + '/' + self.dataset + '/' + lp + '/' + sys
                sys_file_plain = self.dir + '/' + 'plain' + '/' + 'system-outputs' + '/' + self.dataset + '/' + lp + '/' + sys.replace('.out', '')

                extractor = FeatureExtractor()
                extractor.read_config()
                extractor.get_feature_names()
                extractor.extract_features(tgt_plain=sys_file_plain,
                                           ref_plain=ref_file_plain,
                                           tgt_parse=sys_file_parse,
                                           ref_parse=ref_file_parse,
                                           align_dir=self.dir + '/' + 'alignments')

                for i, val in enumerate(extractor.vals):
                    meta_data.write('\t'.join([self.dataset, lp, get_sys_name(sys, self.dataset), str(i)]) + '\n')
                    features.write('\t'.join([str(x) for x in val]) + '\n')

        features.close()
        meta_data.close()

    def learn_to_rank(self, f_features, f_meta_data, f_judgements, f_out_feats, f_out_obj):

        human_ranks = HumanRank()
        human_ranks.add_human_data(f_judgements, self.lps)

        features = read_tsv(f_features)
        meta_data = read_tsv(f_meta_data)

        out_feats = open(f_out_feats, 'w')
        out_obj = open(f_out_obj, 'w')

        for lp in self.lps:

            for case in human_ranks[lp]:
                if case.sign == '=':
                    continue

                winner = self.find_winner(case)
                loser = self.find_loser(case)
                good_features = features[self.get_idx(meta_data, self.dataset, lp, winner, case.phrase)]
                bad_features = features[self.get_idx(meta_data, self.dataset, lp, loser, case.phrase)]

                out_feats.write('\t'.join([str(x) for x in self.get_positive_instance(good_features, bad_features)]) + '\n')
                out_feats.write('\t'.join([str(x) for x in self.get_negative_instance(good_features, bad_features)]) + '\n')
                out_obj.write('1' + '\n')
                out_obj.write('0' + '\n')

    def logistic_evaluate(self, features, coefficients):

        results = []
        for i, phrase in enumerate(features):
            score = 0.0
            for j, feat in enumerate(phrase):
                score += float(feat) * coefficients[0][j]
            results.append(score)

        return results

    @staticmethod
    def logistic_run(x_train_path, x_test_path, y_train_path):

        x_train = read_features_file(x_train_path, '\t')
        y_train = read_reference_file(y_train_path, '\t')
        x_test = read_features_file(x_test_path, '\t')

        estimator = LogisticRegression(penalty='l2',
                                       dual=False,
                                       tol=0.0001,
                                       C=1.0,
                                       fit_intercept=True,
                                       intercept_scaling=1,
                                       class_weight=None,
                                       random_state=None,
                                       solver='liblinear',
                                       max_iter=100,
                                       multi_class='ovr',
                                       verbose=0,
                                       warm_start=False,
                                       n_jobs=1)
        estimator.fit(x_train, y_train)
        y_hat = estimator.predict_proba(x_test)
        coefficients = estimator.coef_

        return [y_hat, coefficients]

    def svc(self):
        predicted = learn_model.run(os.getcwd() + '/' + 'config' + '/' + 'learner' + '/' + 'svc.cfg',
                                x_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2015.features_ltr.tsv',
                                x_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2014.features.tsv',
                                y_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2015.objective_ltr.tsv',
                                y_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2014.objective_dummy.tsv'
                                )


    def get_idx(self, meta_data, dataset, lp, system, phr):
        return meta_data.index([dataset, lp, system, str(phr)])

    def find_winner(self, case):

        if case.sign == '<':
            return case.sys1
        else:
            return case.sys2

    def find_loser(self, case):

        if case.sign == '<':
            return case.sys2
        else:
            return case.sys1

    def get_positive_instance(self, good, bad):

        new_feature_vector = []

        for i, feature in enumerate(good):
            positive_value = float(feature)
            negative_value = float(bad[i])
            new_feature_vector.append(positive_value - negative_value)

        return new_feature_vector

    def get_negative_instance(self, good, bad):

        new_feature_vector = []

        for i, feature in enumerate(good):
            positive_value = float(feature)
            negative_value = float(bad[i])
            new_feature_vector.append(negative_value - positive_value)

        return new_feature_vector

def main():

    data_dir = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt14-data')
    dataset = 'newstest2014'
    lps = ['cs-en']

    f_out_feats = os.getcwd() + '/' + 'test_wmt' + '/' + dataset + '.' + 'features_cobalt_score.tsv'
    f_out_meta_data = os.getcwd() + '/' + 'test_wmt' + '/' + dataset + '.' + 'meta_data.tsv'

    wmt = WMT(data_dir, dataset, lps)
    wmt.process_wmt(f_out_feats, f_out_meta_data)

    f_features = os.getcwd() + '/' + 'test_wmt' + '/' + dataset + '.' + 'features_cobalt_score' + '.tsv'
    f_meta_data = os.getcwd() + '/' + 'test_wmt' + '/' + dataset + '.' + 'meta_data' + '.tsv'
    f_out_feats = os.getcwd() + '/' + 'test_wmt' + '/' + dataset + '.' + 'features_cobalt_score_ltr' + '.tsv'
    f_out_obj = os.getcwd() + '/' + 'test_wmt' + '/' + dataset + '.' + 'objective_ltr' + '.tsv'
    f_ranks = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt15-manual-evaluation/data/wmt15.ces-eng.csv')
    wmt.learn_to_rank(f_features, f_meta_data, f_ranks, f_out_feats, f_out_obj)
    # wmt.svc()

    # f_feature_names = os.getcwd() + '/' + 'config' + '/' + 'features_cobalt_only'
    # x_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/newstest2015.features_cobalt_score_ltr.tsv'
    # x_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/newstest2014.features_cobalt_score.tsv'
    # y_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/newstest2015.objective_ltr.tsv'
    # meta_data_path = '/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/newstest2014.meta_data.tsv'
    # output_path = '/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/word2.scores'
    # arff_path = '/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/newstest2015.features_cobalt_score_ltr.arff'
    # weka_model_out = '/Users/MarinaFomicheva/workspace/upf-cobalt/test_wmt/weka_model.txt'
    # plain_arff(f_feature_names, x_train_path, dataset, arff_path)

    # predictions, coefficients = wmt.logistic_run(x_train_path, x_test_path, y_train_path)
    # coefficents_weka = run(arff_path, weka_model_out)
    # print()
    # meta_data = read_tsv(meta_data_path)
    # features = read_tsv(x_test_path)

    # o = open(output_path, 'w')
    #
    # results = wmt.logistic_evaluate(features, coefficients)
    # for i, val in enumerate(results):
    #     print>>o,  'word2' + '\t' + meta_data[i][1] + '\t' + meta_data[i][0] + '\t' + meta_data[i][2] + '\t' + str(int(meta_data[i][3]) + 1) + '\t' + str(val)
    #
    # o.close()

def read_tsv(my_file):

    phrase = []
    with open(my_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elems = line.strip().split('\t')
            phrase.append(elems)

    return phrase

def set_ref_file(dir_, info_type, dataset, lp):

    if dataset == 'newstest2014':
        if info_type == 'parsed':
            return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-ref.' + lp + '.out'
        elif info_type == 'plain':
            return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-ref.' + lp
    elif dataset == 'newstest2015':
        if info_type == 'parsed':
            return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-' + lp.split('-')[0] + lp.split('-')[1] +\
            '-ref.' + lp.split('-')[1] + '.out'
        elif info_type == 'plain':
            return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-' + lp.split('-')[0] + lp.split('-')[1] +\
            '-ref.' + lp.split('-')[1]

def get_sys_name(sys_id, dataset):
    sys_id = sys_id.replace('.txt', '')
    sys_id = sys_id.replace('.out', '')
    sys_id = sys_id.replace(dataset, '')
    sys_name = '.'.join(sys_id.split('.')[1:-1])
    return sys_name

if __name__ == '__main__':
    main()