__author__ = 'MarinaFomicheva'

import inspect
import numpy as np
import os
from src.features.impl import features as feature_module

class FeatureExtractor(object):

    def __init__(self, cfg):
        self.vals = []
        self.cfg = cfg
        self.feature_names = []

    def extract_features(self, features_to_extract, sents_tgt, sents_ref):

        print("Validating feature names...")

        existing_features = self.get_feature_names(feature_module)
        self.validate_feature_names(features_to_extract, existing_features)

        print("Extracting features...")

        feature_vectors = []

        for name, my_class in sorted(inspect.getmembers(feature_module)):

            if not inspect.isclass(my_class):
                continue

            if not my_class.__module__ == feature_module.__name__:
                continue

            instance = my_class()

            if instance.get_name() not in features_to_extract:
                continue

            print(name)

            feature_vector = []

            for i, sent_tgt in enumerate(sents_tgt):

                instance.run(sent_tgt, sents_ref[i])
                feature_vector.append(instance.get_value())

            feature_vectors.append(feature_vector)

            result = np.array(feature_vectors)
            self.vals = np.transpose(result)

        print("Finished extracting features")

    @staticmethod
    def get_len(my_file):
        return sum(1 for line in open(my_file))

    @staticmethod
    def get_features_from_config_file(config):

        config_features = []
        f_features = open(os.path.expanduser(config.get('Features', 'feature_set')), 'r').readlines()
        for line in sorted(f_features):
            config_features.append(line.strip().split(':')[0])

        return config_features

    @staticmethod
    def get_combinations_from_config_file(config):

        config_features = []
        f_features = open(os.path.expanduser(config.get('Features', 'feature_set')), 'r').readlines()
        for line in sorted(f_features):
            config_features.append(line.strip().split(':')[1])

        return config_features

    @staticmethod
    def get_features_group_name(config):
        file_name = config.get('Features', 'feature_set')
        return file_name.split('/')[-1]

    @staticmethod
    def write_feature_names(feature_names):
        print('\n'.join(feature_names))

    @staticmethod
    def get_feature_names(module):

        my_features = []
        for name, my_class in sorted(inspect.getmembers(module)):

            if not inspect.isclass(my_class):
                continue

            if not my_class.__module__ == module.__name__:
                continue

            instance = my_class()
            my_features.append(instance.get_name())

        return my_features

    @staticmethod
    def get_feature_names_group(module, group):

        my_features = []
        for name, my_class in sorted(inspect.getmembers(module)):

            if not inspect.isclass(my_class):
                continue

            if not my_class.__module__ == module.__name__:
                continue

            instance = my_class()

            if instance.get_group() == group:
                my_features.append(instance.get_name())

        return my_features

    @staticmethod
    def validate_feature_names(features_to_extract, feature_module_names):

        for f in features_to_extract:
            if f not in feature_module_names:
                print("Warning! Feature " + f + "does not exist!")

