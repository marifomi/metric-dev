import os
import fileinput
import codecs
import yaml
import re

from processors.processors import CobaltAligner
from configparser import ConfigParser
from utils.prepare_wmt import PrepareWmt
from features.feature_extractor import FeatureExtractor
from prepare_training_data.ranking_task import RankingTask
from learning.features_file_utils import read_reference_file, split_dataset, split_dataset_repeated_segments, read_features_file
from prepare_training_data.scoring_task import ScoringTask
from processors.processors import POSLanguageModelWordFeatures
from processors.processors import POSTaggerParse
from utils.human_ranking import HumanRanking
from prepare_training_data.evaluate_feature import evaluate_feature_scoring, evaluate_feature_ranking
from utils.human_ranking import HumanRanking


# Ranking Task

config_path = os.getcwd() + '/' + 'config' + '/' + 'wmt.cfg'
config = ConfigParser()
config.readfp(open(config_path))

config_path_learning = os.getcwd() + '/' + 'config/learner/logistic.cfg'
with open(config_path_learning, 'r') as cfg_file:
    config_learning = yaml.load(cfg_file.read())

# ranking_task = RankingTask(config_path)
# ranking_task.prepare_feature_files()

# Training set for learn to rank

# data_structure2 = prepare_wmt.get_data_structure2(config)
# f_judgements = config.get('WMT', 'human_ranking')
# human_rankings = HumanRanking()
# human_rankings.add_human_data(f_judgements, config)
# dataset_for_all = config.get('WMT', 'dataset')
# feature_set_name = os.path.basename(config.get('Features', 'feature_set')).replace(".txt", "")
# feature_values = read_features_file(os.path.expanduser(config.get('WMT', 'output_dir')) + '/' + 'x_' + dataset_for_all + '.' + feature_set_name + '.' + 'all' + '.tsv', "\t")
# ranking_task.training_set_for_learn_to_rank(data_structure2, human_rankings, feature_values)
# ranking_task.train_save(config_learning, config)
# ranking_task.load_get_coefficients(config_learning, config)

# Test learn to rank
# predictions = ranking_task.test_learn_to_rank_coefficients(config_learning, config)
# data_structure = prepare_wmt.get_data_structure(config)
# prepare_wmt.wmt_format(config, feature_set_name, dataset_for_all, predictions, data_structure)


# # ranking_task.training_set_for_learn_to_rank_from_feature_file(config_learning, config)
#
# ranking_task.train_save(config_learning, config)
# predictions = ranking_task.test_learn_to_rank(config_learning)
#
# data_structure = prepare_wmt.get_data_structure2(config)
# prepare_wmt.wmt_format(config, "test", config.get("WMT", "dataset"), predictions, data_structure)


# ranking_task.training_set_for_rank_direct(data_structure, human_rankings, feature_values)

# human_ranking = HumanRanking()
# human_ranking.add_human_data(config.get("WMT", "human_ranking"), config)
# ranking_task.clean_dataset(config_learning, human_ranking)

# input_x = os.getcwd() + '/' + 'test' + '/' + 'x_newstest2014.tsv'
# input_y = os.getcwd() + '/' + 'test' + '/' + 'y_newstest2014.tsv'
# output_dir = os.getcwd() + '/' + 'test'
# split_dataset(input_x, input_y, output_dir)

# gold_labels = read_reference_file(config_learning.get("y_test", None), "\t")
# ranking_task.train_save(config_learning, config)
# predicted = ranking_task.train_predict(config_path_learning)
# predicted = ranking_task.load_predict(config_learning, config)
# ranking_task.evaluate_predicted(predicted, gold_labels)

# ranking_task.train_save(config_learning, config)
# ranking_task.load_get_coefficients(config_learning, config)
# ranking_task.recursive_feature_elimination(config_learning, config)

# Scoring Task

# config_path_learning = os.getcwd() + '/' + 'config/learner/svr.cfg'
# with open(config_path_learning, 'r') as cfg_file:
#      config_learning = yaml.load(cfg_file.read())
#
# config_path = os.getcwd() + '/' + 'config' + '/' + 'absolute.cfg'
# config = ConfigParser()
# config.readfp(open(config_path))
#
# # evaluate_feature_scoring(config, ['meteor'], 'eamt2009', 'es-en', 'system')
#
# scoring_task = ScoringTask(config_path)
# # scoring_task.prepare_wmt16('parse')
# feature_values, human_scores = scoring_task.get_data()
# scoring_task.save_data(feature_values, human_scores)

# input_x = os.path.expanduser('~/Dropbox/informative_features_for_evaluation/data/absolute_scoring/x_mtc4.meteor_comb_min_fluency_features_alignment_quest.tsv')
# input_y = os.path.expanduser('~/Dropbox/informative_features_for_evaluation/data/absolute_scoring/y_mtc4.fluency_features_alignment_quest.tsv')
# input_y = os.path.expanduser('~/Dropbox/workspace/dataSets/mtc4-manual-evaluation/avg_mean.txt')
# output_dir = os.getcwd() + '/' + 'test'
# split_dataset_repeated_segments(input_x, input_y, output_dir, 919)
# #
# gold_labels = read_reference_file(config_learning.get("y_test", None), "\t")
# predicted = scoring_task.train_predict(config_path_learning)
# scoring_task.evaluate_predicted(predicted, gold_labels)
# #
