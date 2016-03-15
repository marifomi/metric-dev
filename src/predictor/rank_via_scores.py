__author__ = 'MarinaFomicheva'

from src.tools.human_rank import HumanRank
from src.tools.wmt_data import WmtData
from src.tools.run_tools import RunTools
from src.features.feature_extractor import FeatureExtractor as FE
from ConfigParser import ConfigParser
import os
from src.utils.learn_to_rank import LearnToRank

class PredictorRankingTask():

    def evaluate_feature(self, config, sample, features_to_extract):

        # No training, score the dataset using a selected feature

        wmt_data = WmtData()
        wmt_data.preprocess(config, sample, 'plain')
        wmt_data.preprocess(config, sample, 'parse')

        tools = RunTools(config)
        sents_tgt, sents_ref = tools.assign_data(sample)

        extractor = FE(config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)
        scores = [x[0] for x in extractor.vals]
        wmt_data.wmt_format(config, scores, wmt_data.get_lp_sizes(), wmt_data.get_lp_systems(), sample)

    def train(self, config, prefix):

        sample = '.train'
        learner = LearnToRank()
        wmt_data = WmtData()
        features_to_extract = FE.get_features_to_test(config)

        fjudge = config.get('WMT', 'ranks' + '_' + sample.replace('.', ''))
        human_ranks = HumanRank()
        human_ranks.add_human_data(fjudge, config)

        wmt_data.preprocess(config, sample, 'plain')
        wmt_data.preprocess(config, sample, 'parse')

        tools = RunTools(config)
        sents_tgt, sents_ref = tools.assign_data(sample)

        extractor = FE(config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)
        wmt_data.add_features(extractor.vals)

        fx = config.get('Data', 'output') + '/' + prefix + '_' + 'features_rank' + sample
        fy = config.get('Data', 'output') + '/' + prefix + '_' + 'objective_rank' + sample
        learner.learn_to_rank(wmt_data, human_ranks, fx, fy)
        learner.logistic_fit(config, fx, fy)

    def test(self, config):

        sample = '.test'
        learner = LearnToRank()
        wmt_data = WmtData()
        features_to_extract = FE.get_features_to_test(config)

        fjudge = config.get('WMT', 'ranks' + '_' + sample.replace('.', ''))
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

        wmt_data.wmt_format(config, [x[1] for x in preds], wmt_data.get_lp_sizes(), wmt_data.get_lp_systems())

def main():

    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/wmt.cfg'))
    predictor = PredictorRankingTask()
    predictor.evaluate_feature(cfg, '.test', ['count_all_oov'])

if __name__ == '__main__':
    main()

