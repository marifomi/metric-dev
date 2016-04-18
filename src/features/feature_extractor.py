import inspect
import numpy as np
import os
import sys
from src.features.impl.features import *
from src.features.impl.features_non_aligned_chunks import *
from src.features.impl.abstract_feature import *

__author__ = 'marina'


class FeatureExtractor(object):

    def __init__(self, cfg):
        self.vals = []
        self.cfg = cfg
        self.feature_names = []

    def extract_features(self, features_to_extract, sents_tgt, sents_ref):
        print("Validating feature names...")

        self.validate_feature_names(features_to_extract, self.get_feature_names())

        print("Extracting features...")

        feature_vectors = []

        for my_class in sorted(list(FeatureExtractor.__iter_subclasses__(AbstractFeature)),
                               key=lambda x: str(x)):

            instance = my_class()

            if str(instance) not in features_to_extract:
                continue

            feature_vector = []

            print("Running " + str(instance))

            for i, sent_tgt in enumerate(sents_tgt):
                instance.run(sent_tgt, sents_ref[i])
                feature_vector.append(instance.get_value())

            feature_vectors.append(feature_vector)

        result = np.array(feature_vectors, dtype=object)
        #self.vals = np.transpose(result)
        self.vals = [list(x) for x in zip(*feature_vectors)]

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
    def get_feature_names():

        my_features = []
        for my_class in sorted(list(FeatureExtractor.__iter_subclasses__(AbstractFeature)),
                               key=lambda x: str(x)):
            my_features.append(str(my_class()))

        return my_features

    @staticmethod
    def validate_feature_names(features_to_extract, feature_module_names):

        for f in features_to_extract:
            if f not in feature_module_names:
                print("Warning! Feature " + f + "does not exist!")

    @staticmethod
    def __iter_subclasses__(cls, _seen=None):

        if not isinstance(cls, type):
            raise TypeError('iter_subclasses must be called with '
                            'new-style classes, not %.100r' % cls)
        if _seen is None:
            _seen = set()
        try:
            subs = cls.__subclasses__()
        except TypeError: # fails only when cls is type
            subs = cls.__subclasses__(cls)
        for sub in subs:
            if sub not in _seen:
                _seen.add(sub)
                yield sub
                for sub in FeatureExtractor.__iter_subclasses__(sub, _seen):
                    yield sub


