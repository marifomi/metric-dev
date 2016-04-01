__author__ = 'MarinaFomicheva'

import numpy as np

from src.tools.prepare_wmt import PrepareWmt
from src.tools.human_ranking import HumanRanking
from src.tools.run_processors import RunProcessors
from src.features.feature_extractor import FeatureExtractor
from src.learning import learn_model
from sklearn.metrics import accuracy_score
from ConfigParser import ConfigParser


class RankingTask(object):

    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.readfp(open(config_path))

    def get_data(self):

        process_wmt = PrepareWmt()
        data_structure = process_wmt.get_data_structure2(self.config.get('WMT', 'input_dir'))
        process_wmt.print_data_set(self.config, data_structure)

        f_judgements = self.config.get('WMT', 'human_ranking')
        human_rankings = HumanRanking()
        human_rankings.add_human_data(f_judgements, self.config)

        process = RunProcessors(self.config)
        sents_tgt, sents_ref = process.run_processors()

        extractor = FeatureExtractor(self.config)
        features_to_extract = FeatureExtractor.get_features_from_config_file(self.config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)

        return data_structure, human_rankings, extractor.vals

    def training_set_for_rank_direct(self, data_structure, human_rankings, feature_values, ignore_ties=True):

        data_set = self.config.get('WMT', 'dataset')
        f_features = open(self.config.get('WMT', 'output_dir') + '/' + 'x_' + data_set, 'w')
        f_objective = open(self.config.get('WMT', 'output_dir') + '/' + 'y_' + data_set, 'w')

        for lang_pair in sorted(human_rankings.keys()):

            for human_comparison in human_rankings[lang_pair]:

                label = self.signs_to_labels(human_comparison.sign, ignore_ties)
                if label is None:
                    continue

                f_objective.write(label + '\n')

                idx_sys1 = self.get_sentence_idx(data_set, lang_pair, data_structure, human_comparison.sys1)
                idx_sys2 = self.get_sentence_idx(data_set, lang_pair, data_structure, human_comparison.sys2)

                difference_vector = np.fabs(np.subtract(feature_values[idx_sys1], feature_values[idx_sys2]))

                f_features.write('\t'.join([str(diff) for diff in difference_vector]) + '\n')

        f_features.close()
        f_objective.close()

    def training_set_for_learn_to_rank(self, data_structure, human_rankings, feature_values):

        data_set = self.config.get('WMT', 'dataset')
        f_features = open(self.config.get('WMT', 'output_dir') + '/' + 'x_' + data_set, 'w')
        f_objective = open(self.config.get('WMT', 'output_dir') + '/' + 'y_' + data_set, 'w')

        for lang_pair in sorted(human_rankings.keys()):

            for human_comparison in human_rankings[lang_pair]:

                if human_comparison.sign == '=':
                    continue

                winner, loser = self.find_winner_loser(human_comparison)

                idx_winner = self.get_sentence_idx(data_set, lang_pair, data_structure, winner)
                idx_loser = self.get_sentence_idx(data_set, lang_pair, data_structure, loser)

                positive_instance, negative_instance = self.get_instance(feature_values[idx_winner],
                                                                         feature_values[idx_loser])

                f_features.write('\t'.join([str(x) for x in positive_instance]) + '\n')
                f_features.write('\t'.join([str(x) for x in negative_instance]) + '\n')

                f_objective.write('1' + '\n')
                f_objective.write('0' + '\n')

        f_features.close()
        f_objective.close()

    @staticmethod
    def predict(config_path):
        predicted = learn_model.run(config_path)
        return predicted

    @staticmethod
    def evaluate_predicted(predicted, gold_class_labels, score='accuracy'):
        if score == 'accuracy':
            print("The accuracy score is " + str(accuracy_score(gold_class_labels, predicted)))
        else:
            print("Error! Unknown type of error metric!")

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
    def get_sentence_idx(data_set, lang_pair, data_structure, human_comparison):

        return data_structure.index([data_set, lang_pair, human_comparison.sys1, human_comparison.phrase]),\
               data_structure.index([data_set, lang_pair, human_comparison.sys2, human_comparison.phrase])

    @staticmethod
    def signs_to_labels(sign, ignore_ties):

        if sign == '<':
            return '2'
        elif sign == '>':
            return '0'
        elif ignore_ties is False and sign == '=':
            return '1'
        else:
            return None