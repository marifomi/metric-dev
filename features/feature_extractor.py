import os

from features.impl.features import *


class FeatureExtractor(object):

    def __init__(self, cfg):
        self.vals = []
        self.cfg = cfg
        self.feature_names = []

    @staticmethod
    def extract_features_static(feature_names, sentences_tgt, sentences_ref):
        print("Validating feature names...")

        FeatureExtractor.validate_feature_names(feature_names, FeatureExtractor.existing_features())

        print("Extracting features...")

        feature_vectors = []

        for my_class in sorted(list(FeatureExtractor.__iter_subclasses__(AbstractFeature)),
                               key=lambda x: str(x)):

            instance = my_class()

            if str(instance) not in feature_names:
                continue

            feature_vector = []

            print("Running " + str(instance))

            for i, sent_tgt in enumerate(sentences_tgt):
                instance.run(sent_tgt, sentences_ref[i])
                feature_vector.append(instance.get_value())

            feature_vectors.append(feature_vector)

        print("Finished extracting features")

        return [list(x) for x in zip(*feature_vectors)]

    def extract_features(self, features_to_extract, sents_tgt, sents_ref):
        print("Validating feature names...")

        self.validate_feature_names(features_to_extract, self.existing_features())

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

        self.vals = [list(x) for x in zip(*feature_vectors)]

        print("Finished extracting features")


    @staticmethod
    def get_feature_names_by_group(group):

        for my_class in sorted(list(FeatureExtractor.__iter_subclasses__(AbstractFeature)),
                                   key=lambda x: str(x)):

            instance = my_class()

            if instance.get_group() != group:
                continue

            print(instance.get_name())

    @staticmethod
    def get_len(my_file):
        return sum(1 for line in open(my_file))

    @staticmethod
    def read_feature_names(cfg):
        features_path = os.path.expanduser(cfg.get('Features', 'feature_set'))
        return [f.strip().split(':')[0] for f in open(features_path).readlines()]

    @staticmethod
    def get_combinations_from_config_file(config):

        config_features = []
        f_features = open(os.path.expanduser(config.get('Features', 'feature_set')), 'r').readlines()
        for line in sorted(f_features):
            config_features.append(line.strip().split(':')[1])

        return config_features

    @staticmethod
    def get_features_from_config_file_unsorted(config):

        config_features = []
        f_features = open(os.path.expanduser(config.get('Features', 'feature_set')), 'r').readlines()
        for line in f_features:
            config_features.append(line.strip().split(':')[0])

        return config_features

    @staticmethod
    def get_combinations_from_config_file_unsorted(config):

        config_features = []
        f_features = open(os.path.expanduser(config.get('Features', 'feature_set')), 'r').readlines()
        for line in f_features:
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
    def existing_features():

        my_features = []
        for my_class in sorted(list(FeatureExtractor.__iter_subclasses__(AbstractFeature)),
                               key=lambda x: str(x)):
            my_features.append(str(my_class()))

        return my_features

    @staticmethod
    def validate_feature_names(features_to_extract, feature_module_names):

        for f in features_to_extract:
            if f not in feature_module_names:
                print("Warning! Feature " + f + " does not exist!")

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
