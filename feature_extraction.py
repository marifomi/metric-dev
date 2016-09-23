import os
from configparser import ConfigParser

from features.feature_extractor import FeatureExtractor
from processors.process import Process
from utils.file_utils import write_feature_file
from utils.human_ranking import HumanRanking
from utils.learn_to_rank import learn_to_rank
from utils.ranking_data import RankingData


def feature_extraction(config_features_path):

    config = ConfigParser()
    config.readfp(open(config_features_path))
    wd = config.get('WMT', 'working_directory')
    if not os.path.exists(wd):
        os.mkdir(wd)

    data = RankingData(config)
    data.read_dataset()

    process = Process(config)
    sentences_tgt, sentences_ref = process.run_processors()

    feature_names = FeatureExtractor.read_feature_names(config)
    feature_values = FeatureExtractor.extract_features_static(feature_names, sentences_tgt, sentences_ref)
    write_feature_file(wd + '/' + 'x' + '_' + data.datasets[0].name + '.tsv', feature_values)

    my_dataset = data.plain[0].dataset
    my_lp = data.plain[0].lp
    f_path = wd + '/' + 'x' + '_' + my_dataset + '_' + my_lp + '.tsv'
    f_file = open(f_path, 'w')

    for i, instance in enumerate(data.plain):
        if instance.dataset == my_dataset and instance.lp == my_lp:
            f_file.write('\t'.join([str(x) for x in feature_values[i]]) + "\n")
        else:
            f_file.close()
            my_dataset = instance.dataset
            my_lp = instance.lp
            f_path = wd + '/' + 'x' + '_' + my_dataset + '_' + my_lp + '.tsv'
            f_file = open(f_path, 'w')

    f_judgements = config.get('WMT', 'human_ranking')
    human_rankings = HumanRanking()
    human_rankings.add_human_data(f_judgements, config)
    human_rankings.get_sentence_ids(data)

    learn_to_rank(feature_values, human_rankings, wd + '/' + 'x_learn_to_rank.tsv', wd + '/' + 'y_learn_to_rank.tsv')

