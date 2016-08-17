import os
import re


def substitute_line_number(line, counter):
    tokens = re.sub(r'^.+(\(.+\):)$\n', r'\1', line)
    return 'Sentence #' + str(counter) + ' ' + tokens + '\n'

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
    if '2014' in dataset or '2015' in dataset or '2016' in dataset:
        sys_file = dataset + '.' + sys_name + '.' + lang_pair

    return sys_dir + '/' + sys_file


def sentences(path):
    return sum(1 for line in open(path))
