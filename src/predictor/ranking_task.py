__author__ = 'MarinaFomicheva'

from src.tools.human_rank import HumanRank
from src.tools.wmt_data import WmtData
from src.tools.run_tools import RunTools
from src.features.feature_extractor import FeatureExtractor as FE
from ConfigParser import ConfigParser
import os
from src.utils.learn_to_rank import LearnToRank
from src.learning.sklearn_utils import read_features_file
from sklearn.externals import joblib


class PredictorRankingTask():

    def train(self, config, prefix):

        sample = 'train'
        learner = LearnToRank()
        wmt_data = WmtData()
        features_to_extract = FE.get_features_to_test(config)

        fjudge = config.get('WMT', 'ranks' + '_' + sample)
        human_ranks = HumanRank()
        human_ranks.add_human_data(fjudge, config)

        wmt_data.preprocess(config, sample, 'plain')
        wmt_data.preprocess(config, sample, 'parse')

        tools = RunTools(config)
        sents_tgt, sents_ref = tools.assign_data(sample)

        extractor = FE(config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)
        wmt_data.add_features(extractor.vals)

        fx = config.get('Data', 'output') + '/' + prefix + '_' + 'features_rank' + '.' + sample
        fy = config.get('Data', 'output') + '/' + prefix + '_' + 'objective_rank' + '.' + sample
        learner.learn_to_rank(wmt_data, human_ranks, fx, fy)
        learner.logistic_run(config, fx, fy)

    def test(self, config):

        sample = 'test'
        learner = LearnToRank()
        wmt_data = WmtData()
        features_to_extract = FE.get_features_to_test(config)

        fjudge = config.get('WMT', 'ranks' + '_' + sample)
        human_ranks = HumanRank()
        human_ranks.add_human_data(fjudge, config)

        wmt_data.preprocess(config, sample, 'plain')
        wmt_data.preprocess(config, sample, 'parse')

        tools = RunTools(config)
        sents_tgt, sents_ref = tools.assign_data(sample)

        extractor = FE(config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)
        wmt_data.add_features(extractor.vals)

        preds = learner.logistic_test(config, extractor.vals)

        wmt_data.wmt_format(config, [x[1] for x in preds])

def main():

    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/system.cfg'))
    predictor = PredictorRankingTask()
    predictor.train(cfg, 'vectors_cbow')
    predictor.test(cfg)

if __name__ == '__main__':
    main()

