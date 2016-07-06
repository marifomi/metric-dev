""" Methods for the data in WMT format """

import os
import re


def get_datasets(data_dir):
    return sorted([dataset for dataset in os.listdir(data_dir + '/' + 'system-outputs') if not dataset.startswith('.')])


def get_lang_pairs(data_dir, dataset):
    lps = os.listdir(os.path.expanduser(data_dir + '/' + 'system-outputs' + '/' + dataset))
    return sorted([lp for lp in lps if not lp.startswith('.') and '-en' in lp])


def get_system_names(data_dir, dataset, lang_pair):
    system_dir = data_dir + '/' + 'system-outputs' + '/' + dataset + '/' + lang_pair
    return sorted([system_name(path, dataset) for path in os.listdir(system_dir) if not path.startswith('.')])


def system_name(system_file, dataset):

    system_file = system_file.replace('.txt', '')
    system_file = system_file.replace(dataset, '')

    if '2013' in dataset:
        return '.'.join(system_file.split('.')[2:])

    if '2014' in dataset or '2015' in dataset or '2016' in dataset:
        return '.'.join(system_file.split('.')[1:-1])


def reference_path(data_dir, dataset, lang_pair):

    ref_dir = data_dir + '/' + 'references'

    if dataset == 'newstest2015' or dataset == 'newsdiscusstest2015' or dataset == 'newstest2016':
        ref_name = dataset + '-' + lang_pair.split('-')[0] + lang_pair.split('-')[1] + '-' + 'ref' + '.' + 'en'
        return '/'.join([ref_dir, dataset]) + '/' + ref_name

    if dataset == 'newstest2014':
        ref_name = dataset + '-' + 'ref' + '.' + lang_pair
        return '/'.join([ref_dir, dataset]) + '/' + ref_name

    if dataset == 'newstest2013':
        ref_name = dataset + '-' + 'ref' + '.' + lang_pair.split('-')[1]
        return ref_dir + '/' + ref_name


def system_path(data_dir, dataset, lang_pair, sys_name):

    sys_dir = data_dir + '/' + 'system-outputs' + '/' + dataset + '/' + lang_pair

    if '2013' in dataset:
        sys_file = dataset + '.' + lang_pair + '.' + sys_name
    else:
        # for 2014, 2015 and 2016 wmt
        sys_file = dataset + '.' + sys_name + '.' + lang_pair

    return sys_dir + '/' + sys_file


def sentences(path):
    return sum(1 for line in open(path))


def phrase_to_index(dataset, lang_pair, sys_name, phrase_number, data):
    return data.plain.index((dataset, lang_pair, sys_name, phrase_number))


def index_to_phrase(index, data):
    return data.plain[index]


def substitute_line_number(line, counter):
    tokens = re.sub(r'^.+(\(.+\):)$\n', r'\1', line)
    return 'Sentence #' + str(counter) + ' ' + tokens + '\n'


def write_wmt_format(output_path, metric, scores, ranking_data):

    with open(output_path, 'w') as o:

        for i, score in enumerate(scores):

            dataset, lang_pair, system_name, phrase = index_to_phrase(i, ranking_data)

            o.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(metric, dataset, lang_pair, system_name, str(phrase), score))


def read_wmt_format(path, lang_pairs):

    data = []

    with open(path, 'r') as input_file:

        for line in input_file.readlines():

            feature, data_set, lang_pair, system_name, seg_id, value = line.strip().split('\t')

            if not '-en' in lang_pair:
                continue

            if lang_pair not in lang_pairs:
                continue

            data.append([feature, data_set, lang_pair, system_name, int(seg_id), float(value)])

        return [x[-1] for x in sorted(data)]
