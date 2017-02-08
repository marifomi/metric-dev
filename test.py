import sys
import os

from configparser import ConfigParser
from utils.ranking_data import RankingData
from utils.human_ranking import HumanRanking
from processors.process import Process
from features.feature_extractor import FeatureExtractor
from utils.wmt import write_wmt_format
from utils.process_semeval import process_semeval
from nltk.corpus import stopwords

# Read configuration file
config = ConfigParser()
config.readfp(open('test.cfg'))

output_path='/home/marina/workspace/metric-dev/output'
if not os.path.exists(output_path):
    raise Exception("Invalid output path")

# Prepare dataset
ranking_data = RankingData(config)
ranking_data.read_dataset(parsed=True)
plain, parsed = ranking_data.generate_sample(100)
ranking_data.write_plain(plain)
ranking_data.write_parsed(parsed)
ranking_data.write_meta_data(plain, 'output/meta.txt')
meta_data = ranking_data.read_meta_data('output/meta.txt')

# Process dataset
process = Process(config)
sentences_target, sentences_reference = process.run_processors()
cobalt_scores = FeatureExtractor.extract_features_static(['cobalt'], sentences_target, sentences_reference)
ranking_data.write_scores_meta(cobalt_scores, meta_data, metric='cobalt', output_path=output_path + '/' + 'cobalt.cs-en.external.scores')
