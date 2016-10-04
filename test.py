import sys
import os


from configparser import ConfigParser
from utils.ranking_data import RankingData
from utils.write_parsed import write_parsed
from utils.human_ranking import HumanRanking
from analysis.analysis import summary_feature_values, difference_feature_values
from processors.process import Process
from features.feature_extractor import FeatureExtractor
from utils.wmt import write_wmt_format
from utils.process_semeval import process_semeval

# Read configuration file
config = ConfigParser()
config.readfp(open('test.cfg'))

# Prepare dataset
ranking_data = RankingData(config)
ranking_data.read_dataset()
ranking_data.write_dataset()
write_parsed(config.get('Data', 'input_dir').replace('plain', 'parse'), config.get('Data', 'working_dir'), ['cs-en'])

# Process dataset
process = Process(config)
sentences_target, sentences_reference = process.run_processors()

# Extract features
cobalt_scores = [x[0] for x in FeatureExtractor.extract_features_static(['cobalt'], sentences_target, sentences_reference)]
write_wmt_format('data_test/cobalt.scores', 'cobalt', cobalt_scores, ranking_data)

sys.exit()

human_ranking = HumanRanking()
human_ranking.add_human_data(config)
human_ranking.clean_data()

difference_feature_values(26,
                         '/home/marina/Dropbox/experiments_fluency/test_learn_to_rank/x_newstest2016.cobalt_comb_min.cs-en.tsv',
                          range1=(20993, 23992), range2=(23992, 26991))



