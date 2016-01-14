__author__ = 'MarinaFomicheva'

import os
from src.utils import sample_dataset
from src.features.feature_extractor import FeatureExtractor
from sklearn import cross_validation as cv
from src.learning import learn_model
import numpy as np


class PredictorAbsoluteScores():

    def __init__(self):
        self.data_size = int
        self.paths = {}

    def set_paths(self):

        self.paths['f_src_plain'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'source.dummy'
        self.paths['f_tgt_plain'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'system'
        self.paths['f_tgt_parse'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'system.parse'
        self.paths['f_ref_plain'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'reference'
        self.paths['f_ref_parse'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'reference.parse'
        self.paths['f_eval'] = '/Users/MarinaFomicheva/Dropbox/workspace/dataSets/wmt13-manual-evaluation/continuous/z-seg-scrs/sample.seg.ad.stnd.all-en.csv'
        self.paths['f_train_phr'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'train_phr.txt'
        self.paths['f_test_phr'] = os.getcwd() + '/' + 'data' + '/' + 'wmt13_graham' + '/' + 'test_phr.txt'
        self.paths['f_config'] = os.getcwd() + '/' + 'config' + '/' + 'learner' + '/' + 'svr.cfg'

        self.paths['out_dir'] = os.getcwd() + '/' + 'results' + '/' + 'wmt13_graham'

    def set_data_size(self, size):
        self.data_size = size

    def run(self, prefix='cobalt_word'):

        self.set_paths()

        # Create train and test samples from dataset

        self.create_samples(1120)

        # Extract features

        for sample_name in ['train', 'test']:
            my_prefix = ''
            if len(prefix) > 0:
                my_prefix = prefix + '_'

            output_path = self.paths['out_dir'] + '/' + my_prefix + 'features' + '.' + sample_name + '.tsv'
            if os.path.exists(output_path):
                print "Feature files already exist!"
                break
            output = open(output_path, 'w')

            extractor = FeatureExtractor()
            extractor.read_config()
            extractor.get_feature_names()
            extractor.extract_features(src_plain=self.paths['f_src_plain'] + '.' + sample_name,
                                       tgt_plain=self.paths['f_tgt_plain'] + '.' + sample_name,
                                       ref_plain=self.paths['f_ref_plain'] + '.' + sample_name,
                                       tgt_parse=self.paths['f_tgt_parse'] + '.' + sample_name,
                                       ref_parse=self.paths['f_ref_parse'] + '.' + sample_name,
                                       align_dir=os.getcwd() + '/' + 'alignments' + '/' + 'wmt13_graham',
                                       aligner='cobalt',
                                       baseline_dir=os.getcwd() + '/' + 'baselines' + '/' + 'wmt13_graham'
            )

            for phr in extractor.vals:
                output.write('\t'.join([str(x) for x in phr]) + '\n')

            output.close()

        # Learn model

        predicted = learn_model.run(self.paths['f_config'],
                                x_train_path=self.paths['out_dir'] + '/' + my_prefix + 'features' + '.' + 'train' + '.tsv',
                                x_test_path=self.paths['out_dir'] + '/' + my_prefix + 'features' + '.' + 'test' + '.tsv',
                                y_train_path=self.paths['f_eval'] + '.' + 'train',
                                y_test_path=self.paths['f_eval'] + '.' + 'test'
                                )

        human_scores = []
        with open(self.paths['f_eval'] + '.' + 'test', 'r') as f:
            lines = f.readlines()
            for line in lines:
                human_scores.append(float(line.strip()))
            f.close()

        print str(np.corrcoef(predicted, human_scores))

        o_pred = open(self.paths['out_dir'] + '/' + prefix + '_' + 'predictions' + '.tsv', 'w')
        for i, pred in enumerate(predicted):
            print str(i + 1) + '\t' + str(pred) + '\t' + str(human_scores[i])
            o_pred.write(str(i + 1) + '\t' + str(pred) + '\t' + str(human_scores[i]) + '\n')

        o_pred.write(str(np.corrcoef(predicted, human_scores)) + '\n')
        o_pred.close()


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

def main():
    predictor = PredictorAbsoluteScores()
    predictor.run()

if __name__ == '__main__':
    main()











