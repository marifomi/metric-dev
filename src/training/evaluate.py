__author__ = 'MarinaFomicheva'

from src.training.feature_extractor import *

class Evaluator(object):

    def __init__(self):
        self.coefficients = {}

    def get_coefficients(self, model_file):
        file = open(model_file, 'r')

        coefficients = {}
        for line in file:
            m = re.match(r'^([a-z_]+)\s+(.+)\n', line)
            if m:
                attr = m.group(1)
                coeff = float(m.group(2))
                coefficients[attr] = coeff
            if 'Odds Ratios...' in line:
                break

        self.coefficients = coefficients

    def evaluate(self, data_file, model_file, output_directory, feature_set):

        features = FeatureExtractor()
        features.get_from_file(data_file)
        self.get_coefficients(model_file)

        output_file = open(output_directory + '/' + feature_set + '.' + '.'.join(model_file.split('.')[1:3]) + '.' + '.'.join(data_file.split('.')[1:3]), 'w')

        for lang_pair in sorted(features.keys()):

            scores = []

            for system, phrase in sorted(features[lang_pair].keys(), key = lambda x: (x[0], x[1])):
                score = 0.0
                for feat in features[lang_pair][system, phrase]:
                    if not feat[0] in self.coefficients.keys():
                        continue
                    score += float(feat[1]) * self.coefficients[feat[0]]
                scores.append(score)
                print >> output_file, '_'.join(model_file.split('.')[1:3]) + '\t' + lang_pair + '\t' + data_file.split('.')[1] + '\t' + system + '\t' + str(phrase) + '\t' + str(score)
