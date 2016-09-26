import os
from configparser import ConfigParser

import yaml

from prepare_training_data.ranking_task import RankingTask
from utils.file_utils import read_features_file
from utils.human_ranking import HumanRanking
from utils.prepare_wmt import PrepareWmt

# Read configuration files

config_path = os.getcwd() + '/' + 'config' + '/' + 'wmt.cfg'
config = ConfigParser()
config.readfp(open(config_path))

config_path_learning = os.getcwd() + '/' + 'config/learner/logistic.cfg'
with open(config_path_learning, 'r') as cfg_file:
    config_learning = yaml.load(cfg_file.read())

# Prepare feature files
# This needs to be done for both training and testing data, changing the names of the datasets in the configuratio file

prepare_wmt = PrepareWmt()
ranking_task = RankingTask(config_path)
ranking_task.prepare_feature_files()

# Create training set for learn to rank
# Comment the above prepare feature files method

dataset_for_all = config.get('WMT', 'dataset')
feature_set_name = os.path.basename(config.get('Features', 'feature_set')).replace(".txt", "")
data_structure2 = prepare_wmt.get_data_structure2(config)

f_judgements = config.get('WMT', 'human_ranking')
human_rankings = HumanRanking()
human_rankings.add_human_data(f_judgements, config)

feature_values = read_features_file(os.path.expanduser(config.get('WMT', 'output_dir')) + '/' + 'x_' + dataset_for_all + '.' + feature_set_name + '.' + 'all' + '.tsv', "\t")

ranking_task.training_set_for_learn_to_rank(data_structure2, human_rankings, feature_values)
ranking_task.train_save(config_learning, config)

# Run the trained model on a the test feature file and produce the output in WMT format

predictions = ranking_task.test_learn_to_rank_coefficients(config_learning, config)
data_structure = prepare_wmt.get_data_structure(config)
prepare_wmt.wmt_format(config, feature_set_name, dataset_for_all, predictions, data_structure)
