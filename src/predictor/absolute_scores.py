__author__ = 'MarinaFomicheva'

import os
from ConfigParser import ConfigParser
from src.utils import sample_dataset
from src.features.feature_extractor import FeatureExtractor
from src.tools.run_tools import RunTools
from sklearn import cross_validation as cv
from src.learning import learn_model
import scipy
from src.tools.wmt_data import WmtData
from collections import defaultdict
# from src.learning.gaussian_processes import regression

class PredictorAbsoluteScores():

    def __init__(self):
        self.data_size = int
        self.paths = {}

    def set_data_size(self, size):
        self.data_size = size

    def feature_analysis(self, prefix, config, features_to_extract):

        x_train_path=config.get('Data', 'output') + '/' + prefix + '_' + 'features' + '.' + 'train' + '.tsv',
        x_test_path=config.get('Data', 'output') + '/' + prefix + '_' + 'features' + '.' + 'test' + '.tsv',
        y_train_path=config.get('Data', 'human') + '.' + 'train',
        y_test_path=config.get('Data', 'human') + '.' + 'test'

        # regression(x_train_path, x_test_path, y_train_path, y_test_path, features_to_extract)
        # Does not work! Python version problem

    def evaluate_feature(self, config, lp, size, system, sample):

        tools = RunTools(config)
        sents_tgt, sents_ref = tools.assign_data(sample)

        features_to_extract = config.get('Features', 'single_feature')
        extractor = FeatureExtractor(config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)
        scores = [x[0] for x in extractor.vals]
        lp_sizes = {}
        lp_sizes[lp] = size

        lp_systems = defaultdict(list)
        lp_systems[lp].append(system)

        WmtData.wmt_format(config, scores, lp_sizes, lp_systems, sample)

    def prepare_data(self, my_prefix, config, features_to_extract):

        for sample_name in ['.train', '.test']:

            output_path = config.get('Data', 'output') + '/' + my_prefix + '_' + 'features' + sample_name + '.tsv'
            if os.path.exists(output_path):
                print("Feature files already exist!")
                break

            tools = RunTools(config)
            sents_tgt, sents_ref = tools.assign_data(sample_name)

            extractor = FeatureExtractor(config)
            extractor.extract_features(features_to_extract, sents_tgt, sents_ref)

            output = open(output_path, 'w')

            for phr in extractor.vals:
                output.write('\t'.join([str(x) for x in phr]) + '\n')

            output.close()


    def learn_model(self, prefix, config):

        predicted = learn_model.run(config.get('Learner', 'path'),
                                x_train_path=config.get('Data', 'output') + '/' + prefix + '_' + 'features' + '.' + 'train' + '.tsv',
                                x_test_path=config.get('Data', 'output') + '/' + prefix + '_' + 'features' + '.' + 'test' + '.tsv',
                                y_train_path=config.get('Data', 'human') + '.' + 'train',
                                y_test_path=config.get('Data', 'human') + '.' + 'test'
                                )

        os.remove(config.get('Data', 'output') + '/' + prefix + '_' + 'features' + '.' + 'train' + '.tsv')
        os.remove(config.get('Data', 'output') + '/' + prefix + '_' + 'features' + '.' + 'test' + '.tsv')

        return predicted

    def create_samples(self, size):

        # Create train and test samples from dataset

        self.set_data_size(size)
        sampled_phrs = cv.train_test_split(range(self.data_size))
        sample_dataset.save_sampled_phrs(sampled_phrs, self.paths['f_train_phr'], self.paths['f_test_phr'])
        sample_dataset.print_sampled_data(self.paths['f_src_plain'], self.paths['f_train_phr'], self.paths['f_test_phr'], format='plain')
        sample_dataset.print_sampled_data(self.paths['f_tgt_plain'], self.paths['f_train_phr'], self.paths['f_test_phr'], format='plain')
        sample_dataset.print_sampled_data(self.paths['f_ref_plain'], self.paths['f_train_phr'], self.paths['f_test_phr'], format='plain')
        sample_dataset.print_sampled_data(self.paths['f_tgt_parse'], self.paths['f_train_phr'], self.paths['f_test_phr'], format='parsed')
        sample_dataset.print_sampled_data(self.paths['f_ref_parse'], self.paths['f_train_phr'], self.paths['f_test_phr'], format='parsed')
        sample_dataset.print_sampled_data(self.paths['f_eval'], self.paths['f_train_phr'], self.paths['f_test_phr'], format='plain')

    @staticmethod
    def get_human_scores(config):
        human_scores = []
        with open(config.get('Data', 'human') + '.' + 'test', 'r') as f:
            lines = f.readlines()
            for line in lines:
                human_scores.append(float(line.strip()))
            f.close()
        return human_scores

    @staticmethod
    def get_features_to_extract(config):
        features_to_extract = []
        file_ = open(config.get('Features', 'feature_set'), 'r')
        for line in file_:
            features_to_extract.append(line.strip())
        return sorted(features_to_extract)

def corr_feature_set(features_to_extract, set_name):

    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/system.cfg'))

    predictor = PredictorAbsoluteScores()
    prefix = set_name
    human_scores = predictor.get_human_scores(cfg)
    predicted = predictor.prepare_data(prefix, cfg, features_to_extract)
    corr = scipy.stats.pearsonr(human_scores, predicted)[1][0]

    return corr


def main():
    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/absolute.cfg'))

    predictor = PredictorAbsoluteScores()
    predictor.evaluate_feature(cfg, 'ch-en', 5514, 'system', '')


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











