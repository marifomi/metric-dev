__author__ = 'MarinaFomicheva'

import codecs
from os import listdir
from os.path import isfile, join
import inspect
from collections import defaultdict

from load_resources import *
import features
from utils.core_nlp_utils import *


class FeatureExtractor(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, dict)

    def extract_features(self, paths, dataset, direction, max_segments, selected_features):

        config = ConfigParser()
        config.readfp(open(paths))

        reference_dir = config.get('Paths', 'reference_dir')
        test_dir = config.get('Paths', 'test_dir')
        ppdbFileName = config.get('Paths', 'ppdbFileName')
        vectorsFileName = config.get('Paths', 'vectorsFileName')

        loadPPDB(ppdbFileName)
        loadWordVectors(vectorsFileName)
        aligner = Aligner('english')

        ref_file_name = self.norm_ref_file_name(reference_dir, dataset, direction)
        ref_data = read_sentences(codecs.open(ref_file_name, encoding='UTF-8'))
        testFiles = [f for f in listdir(test_dir + '/' + dataset + '/' + direction) if isfile(join(test_dir + '/' + dataset + '/' + direction, f))]

        for t in testFiles:
            system = self.get_system_name(t, dataset, direction)
            test_data = read_sentences(codecs.open(test_dir + '/' + dataset + '/' + direction + '/' + t, encoding='UTF-8'))

            for i, phrase_ref in enumerate(ref_data):
                num_phrase = i + 1

                if max_segments != 0 and num_phrase > max_segments:
                    break

                alignments = aligner.align(test_data[i], phrase_ref)
                candidate_parsed = prepareSentence2(test_data[i])
                reference_parsed = prepareSentence2(phrase_ref)
                sentence_features = self.compute_features(test_data[i], phrase_ref, candidate_parsed, reference_parsed, alignments, selected_features)
                self[direction][(system, num_phrase)] = sentence_features
                print direction + ',' + system + ',' + str(num_phrase)

    def print_features(self, output_file, lang_pair):

        for (system, phrase) in self[lang_pair].keys():

            print >>output_file, lang_pair + ',' + system + ',' + str(phrase) + ',' + ','.join([':'.join((x[0], str(x[1]))) for x in self[lang_pair][system, phrase]])

    def get_from_file(self, my_file, selected_features):

        for line in my_file:
            translation = line.strip()
            lang_pair = translation.split(',')[0]
            system = translation.split(',')[1]
            phrase = int(translation.split(',')[2])
            features = translation.split(',')[3:]
            feature_tuples = []
            for feature in features:

                attr = feature.split(':')[0]
                value = float(feature.split(':')[1])

                if len(selected_features) > 0 and attr not in selected_features:
                    continue

                feature_tuple = tuple([attr, value])
                feature_tuples.append(feature_tuple)
            self[lang_pair][(system, phrase)] = feature_tuples
            print lang_pair + ',' + system + ',' + str(phrase)

    def combine(self, feature_extractors):

        segments = []

        for lang_pair in feature_extractors[0].keys():

            for (system, phrase) in feature_extractors[0][lang_pair].keys():
                segments.append((lang_pair, system, phrase))

        for lp, sys, phr in segments:
            combined_features = []

            for feature_extractor in feature_extractors:
                combined_features += feature_extractor[lp][sys, phr]

            self[lp][sys, phr] = combined_features


    @staticmethod
    def read_features_from_file(file_like):

        selected_features = []

        for line in file_like:
            selected_features.append(line.strip())

        return selected_features

    @staticmethod
    def compute_features(candidate, reference, candidate_parsed, reference_parsed, alignments, selected_features):

        feature_vector = []

        for name, my_class in inspect.getmembers(features):

            if 'Abstract' in name:
                continue

            if not inspect.isclass(my_class):
                continue

            instance = my_class()

            if len(selected_features) > 0 and instance.get_name() not in selected_features:
                continue

            instance.run(candidate, reference, candidate_parsed, reference_parsed, alignments)
            feature_vector.append((instance.get_name(), instance.get_value()))

        return feature_vector

    @staticmethod
    def get_feature_names_cobalt():

        names = []

        for name, my_class in inspect.getmembers(features):

            if 'Abstract' in name or not inspect.isclass(my_class):
                continue

            instance = my_class()
            names.append(instance.get_name())
        return names

    @staticmethod
    def get_system_name(test_file, dataset, lang_pair):

        if 'newstest2013' in dataset:
            sys_name = re.sub('^%s\.%s\.(?P<name>.+)\.%s$' % (dataset, lang_pair, 'out'), '\g<name>', test_file)
        elif '2007' in dataset:
            sys_name = test_file.split('.')[0]
        elif 'eamt' in dataset:
            sys_name = 'smt'
        else:
            sys_name = test_file.split('.')[1] + '.' + test_file.split('.')[2]

        return sys_name

    @staticmethod
    def norm_ref_file_name(ref_dir, dataset, lang_pair):

        if 'newstest2013' in dataset:
            file_name = ref_dir + '/' + dataset + '/' + dataset + '-ref.' + lang_pair.split('-')[1] + '.out'
        elif dataset == 'newstest2015' or dataset == 'newsdiscusstest2015':
            file_name = ref_dir + '/' + dataset + '/' + dataset + '-' + lang_pair.split('-')[0] + lang_pair.split('-')[1] + '-ref.' + lang_pair.split('-')[1] + '.out'
        elif '2007' in dataset:
            file_name = ref_dir + '/' + dataset + '/' + dataset + '.' + lang_pair.split('-')[1] + '.out'
        elif 'eamt' in dataset:
            file_name = ref_dir + '/' + dataset + '/' + lang_pair + '/target_postedited.out'
        else:
            file_name = ref_dir + '/' + dataset + '-ref.' + lang_pair + '.out'

        return file_name
