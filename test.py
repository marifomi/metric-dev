import sys
import os

from configparser import ConfigParser
from utils.ranking_data import RankingData
from utils.write_parsed import write_parsed
from utils.human_ranking import HumanRanking
from processors.process import Process
from features.feature_extractor import FeatureExtractor
from utils.wmt import write_wmt_format
from utils.process_semeval import process_semeval
from nltk.corpus import stopwords

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
cobalt_scores = FeatureExtractor.extract_features_static(['cobalt'], sentences_target, sentences_reference)
# print(str(cobalt_scores[-1][0]))
ranking_data.write_scores_wmt_format(cobalt_scores, metric='cobalt', output_path='output/cobalt.scores')
