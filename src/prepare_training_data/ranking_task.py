
from src.utils.prepare_wmt import PrepareWmt
from src.utils.human_ranking import HumanRanking
from src.utils.wmt_kendall_variants import variants_definitions
from src.processors.run_processors import RunProcessors
from src.features.feature_extractor import FeatureExtractor
from src.learning import learn_model
from src.learning.features_file_utils import read_reference_file, read_features_file
from src.learning.features_file_utils import write_reference_file, write_feature_file
from src.learning.learn_model import scale_datasets
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.feature_selection import RFECV, RFE
from sklearn.cross_validation import StratifiedKFold
from configparser import ConfigParser
from json import loads
from collections import defaultdict
from itertools import combinations
import yaml
import os
import numpy as np
import itertools

__author__ = 'MarinaFomicheva'


class RankingTask(object):

    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.readfp(open(os.path.expanduser(config_path)))

    def get_data(self):

        process_wmt = PrepareWmt()
        data_structure1 = process_wmt.get_data_structure(self.config)
        data_structure2 = process_wmt.get_data_structure2(self.config)
        process_wmt.print_data_set(self.config, data_structure1)

        if 'Parse' in loads(self.config.get("Resources", "processors")):
            process_wmt_parse = PrepareWmt(data_type='parse')
            data_structure_parse = process_wmt_parse.get_data_structure(self.config)
            process_wmt_parse.print_data_set(self.config, data_structure_parse)

        f_judgements = self.config.get('WMT', 'human_ranking')
        maximum_comparisons = int(self.config.get('WMT', 'maximum_comparisons'))
        human_rankings = HumanRanking()
        human_rankings.add_human_data(f_judgements, self.config, max_comparisons=maximum_comparisons)

        process = RunProcessors(self.config)
        sents_tgt, sents_ref = process.run_processors()

        extractor = FeatureExtractor(self.config)
        features_to_extract = FeatureExtractor.get_features_from_config_file(self.config)

        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)

        return data_structure2, human_rankings, extractor.vals

    def test_set_for_rank_to_scores(self, data_structure, feature_values, config_path_learning):

        sentences_systems = defaultdict(list)

        combination_methods = FeatureExtractor.get_combinations_from_config_file(self.config)
        data_set_name = self.config.get('WMT', 'dataset')
        f_features = open(os.path.expanduser(self.config.get('WMT', 'output_dir')) + '/' + 'x_' + data_set_name, 'w')
        meta_data = defaultdict(list)

        for data_set, lang_pair, system_name, phrase_number in data_structure:
            sentences_systems[data_set, lang_pair, phrase_number].append(system_name)

        for data_set, lang_pair, phrase_number in sorted(sentences_systems.keys()):

            system_pairs = list(combinations(sentences_systems[data_set, lang_pair, phrase_number], 2))

            for sys1, sys2 in sorted(system_pairs):

                idx_sys1, idx_sys2 = self.get_sentence_idx(data_set, lang_pair, data_structure, phrase_number, sys1, sys2)

                combined_features = []
                for i in range(len(feature_values[0])):
                    combined_feature = self.combine_feature_values(combination_methods[i], feature_values[idx_sys1][i],
                                                                   feature_values[idx_sys2][i])
                    combined_features.append(combined_feature)

                f_features.write('\t'.join([val for val in combined_features]) + '\n')
                meta_data[data_set, lang_pair, phrase_number].append([sys1, sys2])

        f_features.close()

        results = defaultdict(list)
        confidence_scores = self.get_confidence_scores(config_path_learning)
        count = 0
        for data_set, lang_pair, phrase_number in sorted(meta_data.keys()):
            for sys1, sys2 in sorted(meta_data[data_set, lang_pair, phrase_number]):
                results[data_set, lang_pair, phrase_number].append([sys1, sys2, confidence_scores[count]])
                count += 1

        return results

    def convert_rank_to_scores(self, meta_data):

        for data_set, lang_pair, phrase_number in sorted(meta_data.keys()):
            system_names = set(self.flatten([(x[0], x[1]) for x in meta_data[data_set, lang_pair, phrase_number]]))
            for system_name in system_names:
                system_score = 0
                for sys_pair in meta_data[data_set, lang_pair, phrase_number]:
                    if system_name in sys_pair:
                        if sys_pair.index(system_name) == 0:
                            system_score += sys_pair[2]
                        else:
                            system_score -= sys_pair[2]
                print('\t'.join([data_set, lang_pair, system_name, str(phrase_number), str(system_score)]))

    @staticmethod
    def get_confidence_scores(config_path):

        with open(config_path, 'r') as cfg_file:
            config = yaml.load(cfg_file.read())

        learning_config = config.get("learning", None)
        method_name = learning_config.get("method", None)
        x_test = read_features_file(config.get("x_test", None), '\t')
        estimator = joblib.load(config.get("save", None) + '/' + method_name + '.pkl')
        confidence_scores = estimator.decision_function(x_test)
        return confidence_scores

    def training_set_for_rank_direct(self, data_structure, human_rankings, feature_values, ignore_ties=True):

        combination_methods = FeatureExtractor.get_combinations_from_config_file(self.config)
        data_set_name = self.config.get('WMT', 'dataset')
        feature_set_name = os.path.basename(self.config.get('Features', 'feature_set')).replace(".txt", "")

        for dataset, lang_pair in sorted(human_rankings.keys()):

            f_features = open(os.path.expanduser(self.config.get('WMT', 'output_dir')) + '/' + 'x_' + data_set_name + '.' + feature_set_name + '.' + lang_pair + '.tsv', 'w')
            f_objective = open(os.path.expanduser(self.config.get('WMT', 'output_dir')) + '/' + 'y_' + data_set_name + '.' + feature_set_name + '.' + lang_pair + '.tsv', 'w')
            f_meta_data = open(os.path.expanduser(self.config.get('WMT', 'output_dir')) + '/' + 'meta_' + data_set_name + '.' + feature_set_name + '.' + lang_pair + '.tsv', 'w')

            for human_comparison in human_rankings[dataset, lang_pair]:

                label = self.signs_to_labels(human_comparison.sign, ignore_ties=ignore_ties)
                if label is None:
                    continue

                f_objective.write(label + '\n')

                seg_id = human_comparison.phrase
                sys1 = human_comparison.sys1
                sys2 = human_comparison.sys2
                idx_sys1, idx_sys2 = self.get_sentence_idx(dataset, lang_pair, data_structure, seg_id, sys1, sys2)
                f_meta_data.write(str(idx_sys1) + '\t' + str(idx_sys2) + '\n')

                combined_features = []
                for i in range(len(feature_values[0])):
                    combined_feature = self.combine_feature_values(combination_methods[i], feature_values[idx_sys1][i],
                                                                   feature_values[idx_sys2][i])
                    combined_features.append(combined_feature)

                f_features.write('\t'.join([val for val in combined_features]) + '\n')

            f_features.close()
            f_objective.close()
            f_meta_data.close()

    def training_set_for_learn_to_rank(self, data_structure, human_rankings, feature_values):

        data_set_name = self.config.get('WMT', 'dataset')
        f_features = open(os.path.expanduser(self.config.get('WMT', 'output_dir') + '/' + 'x_' + data_set_name), 'w')
        f_objective = open(os.path.expanduser(self.config.get('WMT', 'output_dir') + '/' + 'y_' + data_set_name), 'w')

        for dataset, lang_pair in sorted(human_rankings.keys()):

            for human_comparison in human_rankings[dataset, lang_pair]:

                if human_comparison.sign == '=':
                    continue

                seg_id = human_comparison.phrase
                winner, loser = self.find_winner_loser(human_comparison)
                idx_winner, idx_loser = self.get_sentence_idx(dataset, lang_pair, data_structure, seg_id, winner, loser)

                positive_instance, negative_instance = self.get_instance(feature_values[idx_winner],
                                                                         feature_values[idx_loser])

                f_features.write('\t'.join([str(x) for x in positive_instance]) + '\n')
                f_features.write('\t'.join([str(x) for x in negative_instance]) + '\n')

                f_objective.write('1' + '\n')
                f_objective.write('0' + '\n')

        f_features.close()
        f_objective.close()

    @staticmethod
    def eliminate_ties(human_comparisons):

        result = defaultdict(list)

        for dataset, lang_pair in sorted(human_comparisons.keys()):

            for comparison in human_comparisons[dataset, lang_pair]:

                if comparison.sign == "=":
                    continue

                result[dataset, lang_pair].append(comparison)

        return result

    @staticmethod
    def clean_dataset(config_learning, human_comparisons):

        feature_values = read_features_file(config_learning.get('x_train'), '\t')
        labels = read_reference_file(config_learning.get('y_train'), '\t')
        new_feature_values = []
        new_labels = []
        human_comparisons = RankingTask.eliminate_ties(human_comparisons)
        comparisons_untied_phrases = defaultdict(list)
        comparisons_untied_signs = defaultdict(list)

        deduplicated_phrases, deduplicated_signs = HumanRanking.deduplicate(human_comparisons)

        for dataset, lang_pair in sorted(human_comparisons.keys()):

            for comparison in human_comparisons[dataset, lang_pair]:

                if comparison.sign == "=":
                    continue
                else:
                    comparisons_untied_phrases[dataset, lang_pair].append([comparison.phrase, comparison.sys1, comparison.sys2])
                    comparisons_untied_signs[dataset, lang_pair].append(comparison.sign)

        for dataset, lang_pair in sorted(human_comparisons.keys()):

            for i, comparison in enumerate(comparisons_untied_phrases[dataset, lang_pair]):

                features = feature_values[i]
                label = labels[i]

                if comparison in deduplicated_phrases[dataset, lang_pair]:

                    if deduplicated_signs[dataset, lang_pair][deduplicated_phrases[dataset, lang_pair].index(comparison)] is None:
                        continue

                    label = RankingTask.signs_to_labels(deduplicated_signs[dataset, lang_pair][deduplicated_phrases[dataset, lang_pair].index(comparison)])

                new_feature_values.append(features)
                new_labels.append(label)

        write_feature_file(config_learning.get('x_train') + "." + "clean", new_feature_values)
        write_reference_file(config_learning.get('y_train') + "." + "clean", new_labels)

    def kendall_tau_scores(self, data_structure, human_comparisons, metric_data, variant='wmt14', max_segments=0):

        try:
            coeff_table = variants_definitions[variant]
        except KeyError:
            raise ValueError("There is no definition for %s variant" % variant)

        numerator = 0
        denominator = 0

        for dataset, lang_pair in sorted(human_comparisons.keys()):

            for pairwise_comparison in human_comparisons[dataset, lang_pair]:

                if max_segments != 0 and pairwise_comparison.phrase > max_segments:
                    continue

                sys1_idx, sys2_idx = self.get_sentence_idx(dataset, lang_pair, data_structure,
                                                           pairwise_comparison.phrase, pairwise_comparison.sys1,
                                                           pairwise_comparison.sys2)

                sys1_score = metric_data[sys1_idx]
                sys2_score = metric_data[sys2_idx]

                compare = lambda x, y: '<' if x > y else '>' if x < y else '='
                metric_comparison = compare(sys1_score, sys2_score)
                coeff = coeff_table[pairwise_comparison.sign][metric_comparison]
                if coeff != 'X':
                    numerator += coeff
                    denominator += 1

            return numerator / float(denominator)

    def kendall_tau_direct(self, human_data, metric_data, variant='wmt14'):

        try:
            coeff_table = variants_definitions[variant]
        except KeyError:
            raise ValueError("There is no definition for %s variant" % variant)

        numerator = 0
        denominator = 0

        for i, val in enumerate(human_data):
            human_sign = self.labels_to_signs(val)
            metric_sign = self.labels_to_signs(metric_data[i])

            coeff = coeff_table[human_sign][metric_sign]
            if coeff != 'X':
                numerator += coeff
                denominator += 1

        return numerator / float(denominator)

    @staticmethod
    def combine_feature_values(method, feature_value1, feature_value2):
        if isinstance(feature_value1, list) and isinstance(feature_value2, list):
            result = ''
            for i, v in enumerate(feature_value1):
                if method == 'average':
                    result += str(np.mean([v, feature_value2[i]])) + '\t'
                elif method == 'difference':
                    return str(np.subtract(v, feature_value2[i])) + '\t'
                elif method == 'absolute_difference':
                    return str(np.fabs(np.subtract(v, feature_value2[i]))) + '\t'
                elif method == 'maximum':
                    return str(np.max([v, feature_value2[i]])) + '\t'
                elif method == 'minimum':
                    return str(np.min([v, feature_value2[i]])) + '\t'
                elif method == 'first':
                    return str(v) + '\t'
                elif method == 'both':
                    return str(v) + '\t' + str(feature_value2[i]) + '\t'
            result.strip()
        else:
            if method == 'average':
                return str(np.mean([feature_value1, feature_value2]))
            elif method == 'difference':
                return str(np.subtract(feature_value1, feature_value2))
            elif method == 'absolute_difference':
                return str(np.fabs(np.subtract(feature_value1, feature_value2)))
            elif method == 'maximum':
                return str(np.max([feature_value1, feature_value2]))
            elif method == 'minimum':
                return str(np.min([feature_value1, feature_value2]))
            elif method == 'first':
                return str(feature_value1)
            elif method == 'both':
                return str(feature_value1) + '\t' + str(feature_value2)

    @staticmethod
    def train_predict(config_path_learning):
        predicted = learn_model.run(config_path_learning)
        return predicted

    @staticmethod
    def load_predict(config_learning, config_data):

        learning_config = config_learning.get("learning", None)
        method_name = learning_config.get("method", None)

        x_train = read_features_file(config_learning.get('x_train'), '\t')
        y_train = read_reference_file(config_learning.get('y_train'), '\t')
        x_test = read_features_file(config_learning.get('x_test'), '\t')
        y_test = read_reference_file(config_learning.get('y_test'), '\t')

        scale = config_learning.get("scale", True)

        if scale:
            x_train, x_test = scale_datasets(x_train, x_test)

        estimator = joblib.load(os.path.expanduser(config_data.get("Learner", "models")) + "/" + method_name + ".pkl")
        predictions = estimator.predict(x_test)

        return predictions

    @staticmethod
    def load_get_coefficients(config_learning, config_data):

        feature_names = FeatureExtractor.get_features_from_config_file_unsorted(config_data)
        combination_methods = FeatureExtractor.get_combinations_from_config_file_unsorted(config_data)

        learning_config = config_learning.get("learning", None)
        method_name = learning_config.get("method", None)

        estimator = joblib.load(os.path.expanduser(config_data.get("Learner", "models")) + "/" + method_name + ".pkl")
        coefficients = estimator.coef_

        feature_list = []
        for i, feature_name in enumerate(feature_names):
            if combination_methods[i] == 'both':
                feature_list.append(feature_name)
                feature_list.append(feature_name)
            else:
                feature_list.append(feature_name)

        for i, name in enumerate(feature_list):
            print(name + "\t" + str(coefficients[0][i]))

    @staticmethod
    def train_save(config_learning, config_data):

        learning_config = config_learning.get("learning", None)
        method_name = learning_config.get("method", None)

        x_train = read_features_file(config_learning.get('x_train'), '\t')
        y_train = read_reference_file(config_learning.get('y_train'), '\t')
        x_test = read_features_file(config_learning.get('x_test'), '\t')
        y_test = read_reference_file(config_learning.get('y_test'), '\t')

        scale = config_learning.get("scale", True)

        if scale:
            x_train, x_test = scale_datasets(x_train, x_test)

        estimator, scorers = learn_model.set_learning_method(config_learning, x_train, y_train)

        estimator.fit(x_train, y_train)
        joblib.dump(estimator, os.path.expanduser(config_data.get('Learner', 'models')) + '/' + method_name + '.pkl')

    @staticmethod
    def test_learn_to_rank(config_path):

        with open(config_path, 'r') as cfg_file:
            config = yaml.load(cfg_file.read())

        x_test = read_features_file(config.get('x_test'), '\t')
        estimator = joblib.load(os.path.expanduser(config.get('Learner', 'models')) + '/' + 'logistic.pkl')
        return [x[0] for x in estimator.predict_proba(x_test)]

    @staticmethod
    def recursive_feature_elimination(config_learning, config_data):

        feature_names = FeatureExtractor.get_features_from_config_file_unsorted(config_data)
        combination_methods = FeatureExtractor.get_combinations_from_config_file_unsorted(config_data)

        x_train = read_features_file(config_learning.get('x_train'), '\t')
        y_train = read_reference_file(config_learning.get('y_train'), '\t')
        x_test = read_features_file(config_learning.get('x_test'), '\t')
        estimator, scorers = learn_model.set_learning_method(config_learning, x_train, y_train)

        scale = config_learning.get("scale", True)

        if scale:
            x_train, x_test = scale_datasets(x_train, x_test)

        rfe = RFE(estimator, 10, step=1)
        rfe.fit(x_train, y_train)

        feature_list = []

        for i, feature_name in enumerate(feature_names):
             if combination_methods[i] == 'both':
                 feature_list.append(feature_name)
                 feature_list.append(feature_name)
             else:
                 feature_list.append(feature_name)

        for i, name in enumerate(feature_list):
            print(name + "\t" + str(rfe.ranking_[i]))

        predictions = rfe.predict(x_test)

        return predictions

    @staticmethod
    def recursive_feature_elimination_cv(config_learning, config_data):

        feature_names = FeatureExtractor.get_features_from_config_file_unsorted(config_data)
        combination_methods = FeatureExtractor.get_combinations_from_config_file_unsorted(config_data)

        x_train = read_features_file(config_learning.get('x_train'), '\t')
        y_train = read_reference_file(config_learning.get('y_train'), '\t')
        x_test = read_features_file(config_learning.get('x_test'), '\t')
        estimator, scorers = learn_model.set_learning_method(config_learning, x_train, y_train)

        scale = config_learning.get("scale", True)

        if scale:
            x_train, x_test = scale_datasets(x_train, x_test)

        rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(y_train, 2), scoring='accuracy')
        rfecv.fit(x_train, y_train)

        feature_list = []

        for i, feature_name in enumerate(feature_names):
             if combination_methods[i] == 'both':
                 feature_list.append(feature_name)
                 feature_list.append(feature_name)
             else:
                 feature_list.append(feature_name)

        for i, name in enumerate(feature_list):
            print(name + "\t" + str(rfecv.ranking_[i]))

        predictions = rfecv.predict(x_test)

        return predictions


    def evaluate_predicted(self, predicted, gold_class_labels):
        output_file = open('test_result.out', 'w')
        output_file.write("The accuracy score is " + str(accuracy_score(gold_class_labels, predicted)) + '\n')
        output_file.write("Kendall Tau is " + str(self.kendall_tau_direct(gold_class_labels, predicted, variant='wmt14')) + '\n')
        output_file.close()

    @staticmethod
    def get_instance(winner_feature_values, loser_feature_values):

        positive_instance = np.subtract(winner_feature_values, loser_feature_values)
        negative_instance = np.subtract(loser_feature_values, winner_feature_values)

        return positive_instance, negative_instance

    @staticmethod
    def find_winner_loser(human_comparison):

        if human_comparison.sign == '<':
            return human_comparison.sys1, human_comparison.sys2
        else:
            return human_comparison.sys2, human_comparison.sys1

    @staticmethod
    def get_sentence_idx(data_set, lang_pair, data_structure, seg_id, sys1, sys2):

        return data_structure.index([data_set, lang_pair, sys1, seg_id]),\
               data_structure.index([data_set, lang_pair, sys2, seg_id])

    @staticmethod
    def signs_to_labels(sign, ignore_ties=True):

        if sign == '<':
            return '2'
        elif sign == '>':
            return '0'
        elif ignore_ties is False and sign == '=':
            return '1'
        else:
            return None

    @staticmethod
    def labels_to_signs(sign, ignore_ties=True):

        if sign == 2.0:
            return '<'
        elif sign == 0.0:
            return '>'
        elif ignore_ties is False and sign == 1.0:
            return '='
        else:
            return None

    @staticmethod
    def flatten(list_of_lists):
        "Flatten one level of nesting"
        return list(itertools.chain.from_iterable(list_of_lists))