__author__ = 'MarinaFomicheva'

import os
import scipy
from configparser import ConfigParser
from src.utils import sample_dataset
from src.features.feature_extractor import FeatureExtractor as FE
from src.processors.run_processors import RunProcessors
from sklearn import cross_validation as cv
from src.learning import learn_model
from src.learning.features_file_utils import read_reference_file
from src.features.feature_extractor import FeatureExtractor
from src.features.impl import features_iaa
from src.learning.customize_scorer import pearson_corrcoef


class ScoringTask():

    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.readfp(open(config_path))

    def get_data(self):

        human_scores = read_reference_file(os.path.expanduser(self.config.get('Data', 'human_scores')), '\t')
        process = RunProcessors(self.config)
        sents_tgt, sents_ref = process.run_processors()

        extractor = FeatureExtractor(self.config)
        features_to_extract = FeatureExtractor.get_features_from_config_file(self.config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)

        return extractor.vals, human_scores

    def save_data(self, feature_values, human_scores):

        data_set = self.config.get('Settings', 'dataset')

        f_features = open(os.path.expanduser(self.config.get('Data', 'output_dir')) + '/' + 'x_' + data_set, 'w')
        f_objective = open(os.path.expanduser(self.config.get('Data', 'output_dir')) + '/' + 'y_' + data_set, 'w')

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
    pass


if __name__ == '__main__':
    main()











