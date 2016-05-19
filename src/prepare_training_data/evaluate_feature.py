__author__ = 'MarinaFomicheva'


from src.processors.run_processors import RunProcessors
from src.utils.prepare_wmt import PrepareWmt
from src.features.feature_extractor import FeatureExtractor as FE
from json import loads
import numpy

""" This module scores the data using a set of selected features
    and prints a separate wmt-formatted file for each feature.
    For ranking, wmt dataset is prepared first."""


def evaluate_feature_scoring(config, feature_names, data_set, lang_pair, system_name):

    wmt = PrepareWmt()
    process = RunProcessors(config)
    sents_tgt, sents_ref = process.run_processors()

    for feature in feature_names:
        extractor = FE(config)
        extractor.extract_features([feature], sents_tgt, sents_ref)
        scores = [x[0] for x in extractor.vals]
        wmt.wmt_format_simple(config, feature, data_set, lang_pair, system_name, scores)


def evaluate_feature_ranking(config, features_to_extract):

    if 'Parse' in loads(config.get("Resources", "processors")):
        process_wmt_parse = PrepareWmt(data_type='parse')
        data_structure_parse = process_wmt_parse.get_data_structure(config)
        process_wmt_parse.print_data_set(config, data_structure_parse)

    process_wmt = PrepareWmt()
    data_structure = process_wmt.get_data_structure(config)
    process_wmt.print_data_set(config, data_structure)

    process = RunProcessors(config)
    sents_tgt, sents_ref = process.run_processors()

    for feature in features_to_extract:
        extractor = FE(config)
        extractor.extract_features([feature], sents_tgt, sents_ref)
        scores = [x[0] for x in extractor.vals]
        process_wmt.wmt_format(config, feature, config.get('WMT', 'dataset'), scores, data_structure)

def print_meta_data(feature_data, feature_name, human_rankings, language_pair):

    output = open('test.txt', 'w')

    for pairwise_comparison in human_rankings[language_pair]:
        value_sys1 = feature_data[pairwise_comparison.dataset, language_pair, pairwise_comparison.sys1, pairwise_comparison.phrase][feature_name]
        value_sys2 = feature_data[pairwise_comparison.dataset, language_pair, pairwise_comparison.sys2, pairwise_comparison.phrase][feature_name]
        difference = numpy.fabs(value_sys1 - value_sys2)
        output.write(str(pairwise_comparison.phrase) + '\t' + pairwise_comparison.sys1 + '\t' + pairwise_comparison.sys2 +
              '\t' + pairwise_comparison.sign +
              '\t' + str(value_sys1) +
              '\t' + str(value_sys2) +
              '\t' + str(difference) + '\n'
        )
    output.close()

