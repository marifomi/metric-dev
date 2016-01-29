__author__ = 'MarinaFomicheva'

from ConfigParser import ConfigParser
import os
from src.predictor.absolute_scores import corr_feature_set
from src.features.feature_extractor import FeatureExtractor as FE


def test_feature_sets():

    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/system.cfg'))

    group_name = FE.get_features_group_name(cfg)
    features_to_test = FE.get_features_to_test(cfg)

    if os.path.exists(cfg.get('Data', 'output') + '/' + group_name + '.' + 'summary'):
        "Path exists!"
        return

    output_file = open(cfg.get('Data', 'output') + '/' + group_name + '.' + 'summary', 'w')

    name0 = group_name + '_' + 'all'
    corr0 = corr_feature_set(features_to_test, name0)
    output_file.write(name0 + '\t' + str(corr0) + '\n')

    for feat in features_to_test:

        name1 = group_name + '_' + feat + '_' + 'only'
        corr1 = corr_feature_set(feat, name1)
        output_file.write(name1 + '\t' + str(corr1) + '\n')

        name2 = group_name + '_' + feat + '_' + 'excluded'
        excluding = []

        for ffeat in features_to_test:
            if ffeat == feat:
                continue
            excluding.append(ffeat)

        corr2 = corr_feature_set(excluding, name2)
        output_file.write(name2 + '\t' + str(corr2) + '\n')

    output_file.close()


if __name__ == '__main__':
    test_feature_sets()