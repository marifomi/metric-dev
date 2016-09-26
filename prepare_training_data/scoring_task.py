import codecs
import os
import re
from configparser import ConfigParser
from json import loads

import yaml
from sklearn.feature_selection import RFE

from features.feature_extractor import FeatureExtractor
from learning import learn_model
from learning.customize_scorer import pearson_corrcoef
from learning.learn_model import scale_datasets
from processors.process import Process
from utils.file_utils import read_reference_file, write_lines_to_file, read_features_file, write_feature_file, write_reference_file


class ScoringTask():

    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.readfp(open(config_path))

    @staticmethod
    def substitute_line_number(line, counter):
        tokens = re.sub(r'^.+(\(.+\):)$\n', r'\1', line)
        return 'Sentence #' + str(counter) + ' ' + tokens + '\n'

    def prepare_wmt16(self, data_type='plain'):

        dataset = self.config.get("Settings", "dataset")

        source = []
        target = []
        reference = []
        human = []

        counter_tgt = 0
        counter_ref = 0

        for lp in loads(self.config.get("WMT16", "lang_pairs")):

            if data_type == 'parse':
                input_tgt = codecs.open(os.path.expanduser(self.config.get("WMT16", "input_dir")) + "/" + data_type + "/" + lp + "/" + dataset + "." + "mt-system" + "." + lp + "." + "out", "r", "utf-8").readlines()
                input_ref = codecs.open(os.path.expanduser(self.config.get("WMT16", "input_dir")) + "/" + data_type + "/" + lp + "/" + dataset + "." + "reference" + "." + lp + "." + "out", "r", "utf-8").readlines()
            else:
                input_tgt = codecs.open(os.path.expanduser(self.config.get("WMT16", "input_dir")) + "/" + data_type + "/" + lp + "/" + dataset + "." + "mt-system" + "." + lp, "r", "utf-8").readlines()
                input_ref = codecs.open(os.path.expanduser(self.config.get("WMT16", "input_dir")) + "/" + data_type + "/" + lp + "/" + dataset + "." + "reference" + "." + lp, "r", "utf-8").readlines()
                input_human = codecs.open(os.path.expanduser(self.config.get("WMT16", "input_dir")) + "/" + data_type + "/" + lp + "/" + dataset + "." + "human" + "." + lp, "r", "utf-8").readlines()
                input_src = codecs.open(os.path.expanduser(self.config.get("WMT16", "input_dir")) + "/" + data_type + "/" + lp + "/" + dataset + "." + "source" + "." + lp, "r", "utf-8").readlines()

                for line in input_src:
                    source.append(line)
                for line in input_human:
                    human.append(line)

            for line in input_tgt:

                if data_type == 'parse' and line.startswith('Sentence #'):
                    counter_tgt += 1
                    target.append(ScoringTask.substitute_line_number(line, counter_tgt))
                else:
                    target.append(line)

            for line in input_ref:

                if data_type == 'parse' and line.startswith('Sentence #'):
                    counter_ref += 1
                    reference.append(ScoringTask.substitute_line_number(line, counter_ref))
                else:
                    reference.append(line)

        if data_type == 'parse':
            write_lines_to_file(os.path.expanduser(self.config.get("WMT16", "output_dir")) + "/" + "tgt" + "." + "txt" + ".parse", target)
            write_lines_to_file(os.path.expanduser(self.config.get("WMT16", "output_dir")) + "/" + "ref" + "." + "txt" + ".parse", reference)
        else:
            write_lines_to_file(os.path.expanduser(self.config.get("WMT16", "output_dir")) + "/" + "tgt" + "." + "txt", target)
            write_lines_to_file(os.path.expanduser(self.config.get("WMT16", "output_dir")) + "/" + "ref" + "." + "txt", reference)
            write_lines_to_file(os.path.expanduser(self.config.get("WMT16", "output_dir")) + "/" + "src" + "." + "txt", source)
            write_lines_to_file(os.path.expanduser(self.config.get("WMT16", "output_dir")) + "/" + "human" + "." + "txt", human)

    def get_data(self):

        human_scores = read_reference_file(os.path.expanduser(self.config.get('Data', 'human_scores')), '\t')
        process = Process(self.config)
        sents_tgt, sents_ref = process.run_processors()

        extractor = FeatureExtractor(self.config)
        features_to_extract = FeatureExtractor.read_feature_names(self.config)
        extractor.extract_features(features_to_extract, sents_tgt, sents_ref)

        return extractor.vals, human_scores

    def save_data(self, feature_values, human_scores):

        data_set = self.config.get('Settings', 'dataset')

        # f_features = open(os.path.expanduser(self.config.get('Data', 'output_dir')) + '/' + 'x_' + data_set + '.tsv', 'w')
        # f_objective = open(os.path.expanduser(self.config.get('Data', 'output_dir')) + '/' + 'y_' + data_set + '.tsv', 'w')

        f_features = open(os.path.expanduser(self.config.get('Data', 'output_dir'))
                          + '/' + 'x_' + data_set
                          + '.' + self.config.get("Features", "feature_set").split('/')[-1].replace('.txt', '')
                          + '.' + self.config.get("WMT16", "lang_pair") + '.tsv', 'w')

        f_objective = open(os.path.expanduser(self.config.get('Data', 'output_dir'))
                           + '/' + 'y_' + data_set
                           + '.' + self.config.get("Features", "feature_set").split('/')[-1].replace('.txt', '')
                           + '.' + self.config.get("WMT16", "lang_pair") + '.tsv', 'w')

        for i, score in enumerate(human_scores):
            f_objective.write(str(score) + '\n')
            f_features.write('\t'.join([str(value) for value in feature_values[i]]) + '\n')

        f_features.close()
        f_objective.close()

    @staticmethod
    def get_train_lps(lps, test_lp):

        results = []
        for lp in lps:
            if lp == test_lp:
                continue
            results.append(lp)
        return results

    def round_robin(self, config_path_learning, config_path, feature_set, lps):

        config = ConfigParser()
        config.readfp(open(config_path))

        with open(config_path_learning, "r") as cfg_file:
            config_learning = yaml.load(cfg_file.read())

        f_results = open("results.txt", "w")

        for test_lp in sorted(lps):

            x_train = os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + "x_" + self.config.get("Settings", "dataset") + "." + "train" + "." + "tsv"
            y_train = os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + "y_" + self.config.get("Settings", "dataset") + "." + "train" + "." + "tsv"
            x_test = os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + "x_" + self.config.get("Settings", "dataset") + "." + "test" + "." + "tsv"
            y_test = os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + "y_" + self.config.get("Settings", "dataset") + "." + "test" + "." + "tsv"

            train_lps = ScoringTask.get_train_lps(lps, test_lp)

            train_feature_values = []
            train_reference_values = []

            test_feature_values = read_features_file(os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + \
                                                    "x_" + self.config.get("Settings", "dataset") + "." + feature_set + "." + test_lp + "." + "tsv", "\t")
            test_reference_values = read_reference_file(os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + \
                                                    "y_" + self.config.get("Settings", "dataset") + "." + feature_set + "." + test_lp + "." + "tsv", "\t")

            for train_lp in sorted(train_lps):
                feature_values = read_features_file(os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + \
                                                    "x_" + self.config.get("Settings", "dataset") + "." + feature_set + "." + train_lp + "." + "tsv", "\t")
                reference_values = read_reference_file(os.path.expanduser(self.config.get('Data', 'output_dir')) + "/" + \
                                                    "y_" + self.config.get("Settings", "dataset") + "." + feature_set + "." + train_lp + "." + "tsv", "\t")

                train_feature_values += list(feature_values)
                train_reference_values += list(reference_values)

            write_feature_file(x_train, train_feature_values)
            write_reference_file(y_train, train_reference_values)
            write_feature_file(x_test, test_feature_values)
            write_reference_file(y_test, test_reference_values)

            gold_standard = test_reference_values
            predictions = ScoringTask.train_predict(config_path_learning)
            # predictions = ScoringTask.recursive_feature_elimination(config_learning, config, 50)

            correlation = ScoringTask.evaluate_predicted(predictions, gold_standard)

            f_results.write(test_lp + " " + str(correlation) + " with " + feature_set + "\n")
            os.remove(x_train)
            os.remove(x_test)
            os.remove(y_train)
            os.remove(y_test)

    @staticmethod
    def train_predict(config_path):
        predicted = learn_model.run(config_path)
        return predicted

    @staticmethod
    def evaluate_predicted(predicted, gold_class_labels, score='pearson'):
        if score == 'pearson':
            print("Pearson correlation is " + str(pearson_corrcoef(gold_class_labels, predicted)))
        else:
            print("Error! Unknown type of error metric!")

        return pearson_corrcoef(gold_class_labels, predicted)

    @staticmethod
    def get_human_scores(config):
        human_scores = []
        with open(config.get('Data', 'human'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                human_scores.append(float(line.strip()))
            f.close()
        return human_scores

    @staticmethod
    def recursive_feature_elimination(config_learning, config_data, number_features):

        output = open(os.path.expanduser(config_data.get("Learner", "models")) + "/" + "feature_ranks.txt", "w")

        feature_names = FeatureExtractor.get_combinations_from_config_file_unsorted(config_data)

        x_train = read_features_file(config_learning.get('x_train'), '\t')
        y_train = read_reference_file(config_learning.get('y_train'), '\t')
        x_test = read_features_file(config_learning.get('x_test'), '\t')
        estimator, scorers = learn_model.set_learning_method(config_learning, x_train, y_train)

        scale = config_learning.get("scale", True)

        if scale:
            x_train, x_test = scale_datasets(x_train, x_test)

        rfe = RFE(estimator, number_features, step=1)
        rfe.fit(x_train, y_train)

        for i, name in enumerate(feature_names):
            output.write(name + "\t" + str(rfe.ranking_[i]) + "\n")
            print(name + "\t" + str(rfe.ranking_[i]))

        predictions = rfe.predict(x_test)

        output.close()

        return predictions









