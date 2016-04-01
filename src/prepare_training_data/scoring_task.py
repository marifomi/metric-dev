__author__ = 'MarinaFomicheva'

import os
import scipy
from ConfigParser import ConfigParser
from src.utils import sample_dataset
from src.features.feature_extractor import FeatureExtractor as FE
from src.tools.run_processors import RunProcessors
from sklearn import cross_validation as cv
from src.learning import learn_model
from src.learning.features_file_utils import read_reference_file
from src.features.feature_extractor import FeatureExtractor
from src.features.impl import features_iaa
from src.learning.customize_scorer import pearson_corrcoef


class PredictorAbsoluteScores():

    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.readfp(open(config_path))

    def get_data(self):

        human_scores = read_reference_file(self.config.get('Data', 'human_scores'), '\t')
        process = RunProcessors(self.config)
        sents_tgt, sents_ref = process.run_processors()

        extractor = FeatureExtractor(self.config)
        features_to_extract = FeatureExtractor.get_features_from_config_file(self.config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)

        return extractor.vals, human_scores

    def save_data(self, feature_values, human_scores):

        data_set = self.config.get('Settings', 'dataset')

        f_features = open(self.config.get('WMT', 'output_dir') + '/' + 'x_' + data_set, 'w')
        f_objective = open(self.config.get('WMT', 'output_dir') + '/' + 'y_' + data_set, 'w')

        for i, score in enumerate(human_scores):
            f_objective.write(str(score) + '\n')
            f_features.write('\t'.join([str(value) for value in feature_values[i]]) + '\n')

        f_features.close()
        f_objective.close()

    @staticmethod
    def predict(config_path):
        predicted = learn_model.run(config_path)
        return predicted

    @staticmethod
    def evaluate_predicted(predicted, gold_class_labels, score='pearson'):
        if score == 'pearson':
            print("Pearson correlation is " + str(pearson_corrcoef(gold_class_labels, predicted)))
        else:
            print("Error! Unknown type of error metric!")

    @staticmethod
    def get_human_scores(config):
        human_scores = []
        with open(config.get('Data', 'human'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                human_scores.append(float(line.strip()))
            f.close()
        return human_scores

def main():
    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/absolute.cfg'))

    predictor = PredictorAbsoluteScores()
    feature_names = FE.get_feature_names(features_iaa)
    data_set = 'mtc4'
    lang_pair = 'ch-en'
    system_name = 'system'
    predictor.evaluate_feature(cfg, feature_names, data_set, lang_pair, system_name)


# def main():
#
#     cfg = ConfigParser()
#     cfg.readfp(open(os.getcwd() + '/config/multi_ref.cfg'))
#
#     predictor = PredictorAbsoluteScores()
#     features_to_extract = predictor.get_features_to_extract(cfg)
#     prefix = 'quest_svm_human'
#     predictor.prepare_data(prefix, cfg, features_to_extract)
#     predicted = predictor.learn_model(prefix, cfg)
#
#     human_scores = predictor.get_human_scores(cfg)
#     corr = scipy.stats.pearsonr(human_scores, predicted)
#     print(str(corr))
#
#     o_pred = open(cfg.get('Data', 'output') + '/' + prefix + '_' + 'predictions' + '.tsv', 'w')
#     for i, pred in enumerate(predicted):
#         print(str(i + 1) + '\t' + str(pred) + '\t' + str(human_scores[i]))
#         o_pred.write(str(i + 1) + '\t' + str(pred) + '\t' + str(human_scores[i]) + '\n')
#
#     o_pred.write(str(scipy.stats.pearsonr(predicted, human_scores)) + '\n')
#     o_pred.close()

if __name__ == '__main__':
    main()











