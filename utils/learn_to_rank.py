import numpy as np

from utils.file_utils import write_feature_file, write_reference_file


def learn_to_rank(feature_values, human_comparisons, path_x, path_y):

    xs = []
    ys = []

    for dataset, lp in sorted(human_comparisons.keys()):
        for comparison in human_comparisons[dataset, lp]:

            if comparison.sign == '=':
                continue

            idx_winner, idx_loser = find_winner_loser_index(comparison)
            xs.append(make_instance(feature_values[idx_winner], feature_values[idx_loser]))
            xs.append(make_instance(feature_values[idx_loser], feature_values[idx_winner]))
            ys.append(1)
            ys.append(0)

    write_feature_file(path_x, xs)
    write_reference_file(path_y, ys)


def make_instance(feature_values1, feature_values2):
    return np.subtract(feature_values1, feature_values2)


def find_winner_loser_index(comparison):

    if comparison.sign == '<':
        return comparison.idx_phrase_sys1, comparison.idx_phrase_sys2
    else:
        return comparison.idx_phrase_sys2, comparison.idx_phrase_sys1

