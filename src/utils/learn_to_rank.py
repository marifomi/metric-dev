__author__ = 'MarinaFomicheva'

import numpy as np
from src.learning.sklearn_utils import read_features_file
from src.learning.sklearn_utils import read_reference_file
from src.learning import learn_model
from sklearn.linear_model import LogisticRegression
import os
from sklearn.externals import joblib

class LearnToRank(object):

    def get_idx_simple(self, elem, my_list):

        result = None
        for i, item in enumerate(my_list):
            if item == elem:
                result = i

        if result is None:
            print "Element not found!"
            return
        else:
            return result

    def get_idx_complex(self, wmt_data, lp_idx, sys_idx, phrase):

        lps = sorted(wmt_data.lp_sizes.keys())

        temp_sum = 0
        for i in range(lp_idx):
            temp_sum = np.sum(wmt_data.lp_sizes[lps[i]] * len(wmt_data.lp_systems[lps[i]]))

        return temp_sum + wmt_data.lp_sizes[lps[lp_idx]] * sys_idx + phrase


    def learn_to_rank(self, wmt_data, ranks, fx, fy):

        features = open(fx, 'w')
        objective = open(fy, 'w')

        lps = sorted(wmt_data.lp_sizes.keys())

        for i, lp in enumerate(lps):

            systems = sorted(wmt_data.lp_systems[lp])


            print str(len([rank for rank in ranks[lp] if rank.sign is not '=']))

            cnt = 0
            for j, comp in enumerate(ranks[lp]):

                if comp.sign == '=':
                    continue

                print str(cnt)
                cnt += 1

                winner_system_idx = self.find_winner(comp, systems)
                loser_system_idx = self.find_loser(comp, systems)

                winner_phrase = self.get_idx_complex(wmt_data, i, winner_system_idx, comp.phrase)
                loser_phrase = self.get_idx_complex(wmt_data, i, loser_system_idx, comp.phrase)

                winner_features = wmt_data.features[winner_phrase]
                loser_features = wmt_data.features[loser_phrase]

                features.write('\t'.join([str(x) for x in self.get_positive_instance(winner_features, loser_features)]) + '\n')
                features.write('\t'.join([str(x) for x in self.get_negative_instance(winner_features, loser_features)]) + '\n')
                objective.write('1' + '\n')
                objective.write('0' + '\n')

        features.close()
        objective.close()

    def logistic_test(self, config, features):

        estimator = joblib.load(config.get('Learner', 'models') + '/' + 'logistic.pkl')
        return estimator.predict_proba(features)

    def logistic_coef(self, config):

        estimator = joblib.load(config.get('Learner', 'models') + '/' + 'logistic.pkl')
        return estimator.coef_

    def logistic_run(self, config, x_train_path, y_train_path):

        x_train = read_features_file(x_train_path, '\t')
        y_train = read_reference_file(y_train_path, '\t')
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
        joblib.dump(estimator, config.get('Learner', 'models') + '/' + 'logistic.pkl')

    def svc(self):
        predicted = learn_model.run(os.getcwd() + '/' + 'config' + '/' + 'learner' + '/' + 'svc.cfg',
                                x_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2015.features_ltr.tsv',
                                x_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2014.features.tsv',
                                y_train_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2015.objective_ltr.tsv',
                                y_test_path='/Users/MarinaFomicheva/workspace/upf-cobalt/test/newstest2014.objective_dummy.tsv'
                                )


    def find_winner(self, case, systems):

        idx_sys1 = self.get_idx_simple(case.sys1, systems)
        idx_sys2 = self.get_idx_simple(case.sys2, systems)

        if case.sign == '<':
            return idx_sys1
        else:
            return idx_sys2

    def find_loser(self, case, systems):

        idx_sys1 = self.get_idx_simple(case.sys1, systems)
        idx_sys2 = self.get_idx_simple(case.sys2, systems)

        if case.sign == '<':
            return idx_sys2
        else:
            return idx_sys1

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
