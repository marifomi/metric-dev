__author__ = 'MarinaFomicheva'

import inspect
from src.features.impl import features

class FeatureExtractor(object):

    def __init__(self, cfg):
        self.vals = []
        self.cfg = cfg
        self.feature_names = []

    def extract_features(self, features_to_extract, sents_tgt, sents_ref):

        print "Extracting features"

        for i, sent_tgt in enumerate(sents_tgt):

            sent_feats = []

            for name, my_class in sorted(inspect.getmembers(features)):

                if 'Abstract' in name or 'Scorer' in name:
                    continue

                if not inspect.isclass(my_class):
                    continue

                instance = my_class()

                if instance.get_name() not in features_to_extract:
                    continue

                instance.run(sent_tgt, sents_ref[i])
                print instance.get_name()

                sent_feats.append(instance.get_value())

            self.vals.append(sent_feats)

        print "Finished extracting features"

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


def main():

    for name, my_class in sorted(inspect.getmembers(features)):

        if 'Abstract' in name or 'Scorer' in name:
            continue

        if not inspect.isclass(my_class):
            continue

        instance = my_class()
        print instance.get_name()

if __name__ == '__main__':
    main()