__author__ = 'MarinaFomicheva'

import inspect
from src.features.impl import features
from ConfigParser import ConfigParser
import os
from src.tools.wmt_data import WmtData
from src.tools.human_rank import HumanRank
from src.tools.run_tools import RunTools

class FeatureExtractor(object):

    def __init__(self, cfg):
        self.vals = []
        self.cfg = cfg
        self.feature_names = []

    def extract_features(self, features_to_extract, sents_tgt, sents_ref):

        print("Validating feature names...")

        validate_feature_names(features_to_extract)

        print("Extracting features...")

        for i, sent_tgt in enumerate(sents_tgt):

            sent_feats = []

            for name, my_class in sorted(inspect.getmembers(features)):

                # if 'Abstract' in name or 'Scorer' in name:
                #     continue

                if not inspect.isclass(my_class):
                    continue

                if not my_class.__module__ == 'src.features.impl.features':
                    continue

                instance = my_class()

                if instance.get_name() not in features_to_extract:
                    continue

                instance.run(sent_tgt, sents_ref[i])
                sent_feats.append(instance.get_value())

            self.vals.append(sent_feats)

        print("Finished extracting features")

    @staticmethod
    def get_len(my_file):
        return sum(1 for line in open(my_file))

    @staticmethod
    def get_features_to_test(config):
        features_to_extract = []
        file_ = open(config.get('Features', 'feature_set'), 'r')
        for line in file_:
            features_to_extract.append(line.strip())
        return features_to_extract

    @staticmethod
    def get_features_group_name(config):
        file_name = config.get('Features', 'feature_set')
        return file_name.split('/')[-1]


def write_feature_names():

    features = get_feature_names()
    print('\n'.join(features))


def get_feature_names():

    my_features = []
    for name, my_class in sorted(inspect.getmembers(features)):

        if not inspect.isclass(my_class):
            continue

        if not my_class.__module__ == 'src.features.impl.features':
            continue

        # if 'Abstract' in name or 'Scorer' in name:
        #     continue
        #

        instance = my_class()
        my_features.append(instance.get_name())

    return my_features

def validate_feature_names(features_to_extract):

    existing_features = get_feature_names()

    for f in features_to_extract:
        if f not in existing_features:
            print "Warning! Feature " + f + "does not fecking exist!"


def main():

    config = ConfigParser()
    config.readfp(open(os.getcwd() + '/config/system.cfg'))
    sample = 'test'
    wmt_data = WmtData()

    features_to_extract = FeatureExtractor.get_features_to_test(config)

    fjudge = config.get('WMT', 'ranks' + '_' + sample)
    human_ranks = HumanRank()
    human_ranks.add_human_data(fjudge, config)

    wmt_data.preprocess(config, sample, 'plain')
    wmt_data.preprocess(config, sample, 'parse')

    tools = RunTools(config)
    sents_tgt, sents_ref = tools.assign_data(sample)

    extractor = FeatureExtractor(config)
    extractor.extract_features(features_to_extract, sents_tgt, sents_ref)
    wmt_data.add_features(extractor.vals)

    wmt_data.wmt_format(config, [x[0] for x in extractor.vals], wmt_data.get_lp_sizes(), wmt_data.get_lp_systems())

if __name__ == '__main__':
    config = ConfigParser()
    config.readfp(open(os.getcwd() + '/config/absolute.cfg'))
    features_to_extract = FeatureExtractor.get_features_to_test(config)
    validate_feature_names(features_to_extract)