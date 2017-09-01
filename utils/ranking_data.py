"""This class stores wmt-style ranking data (references and MT outputs) with dataset, language pairs and system ids"""

import codecs
import random

from os.path import expanduser as usr
from utils import wmt
from collections import defaultdict
from json import loads
from collections import namedtuple


class Dataset(object):

    def __init__(self, name):
        self.name = name
        self.system_names = defaultdict(list)
        self.number_sentences = {}


class RankingData(object):
    sentence_tuple = namedtuple("SentenceTuple", ["dataset", "lp", "system", "sentence_num", "target_sentence", "reference_sentence"])

    def __init__(self, config):
        self.cfg = config
        self.dir = self.cfg.get('Data', 'input_dir')
        self.datasets = []
        self.plain = []
        self.parsed = []

    def read_dataset(self, parsed=False):

        dataset_names = wmt.get_datasets(self.dir)
        for dataset_name in dataset_names:
            dataset = Dataset(dataset_name)
            lang_pairs = wmt.get_lang_pairs(self.dir, dataset_name)

            for lp in lang_pairs:
                if len(loads(self.cfg.get('Settings', 'language_pairs'))) > 0 and lp not in loads(self.cfg.get('Settings', 'language_pairs')):
                    continue

                if self.cfg.has_option('Settings', 'system_names'):
                    system_names = loads(self.cfg.get('Settings', 'system_names'))
                else:
                    system_names = wmt.get_system_names(self.dir, dataset_name, lp)

                number_sentences = wmt.sentences(wmt.reference_path(self.dir, dataset.name, lp))

                dataset.system_names[lp] = system_names
                dataset.number_sentences[lp] = number_sentences

                with codecs.open(wmt.reference_path(self.dir, dataset.name, lp), 'r', 'utf8') as ir:
                    reference_sentences = ir.readlines()
                if parsed:
                    with codecs.open(wmt.reference_path(self.dir, dataset.name, lp).replace('plain', 'parse') + '.out', 'r', 'utf8') as pr:
                        reference_parsed = self.read_parsed(pr)

                for system_name in system_names:
                    with codecs.open(wmt.system_path(self.dir, dataset.name, lp, system_name), 'r', 'utf8') as it:
                        target_sentences = it.readlines()
                    if parsed:
                        with codecs.open(wmt.system_path(self.dir, dataset.name, lp, system_name).replace('plain', 'parse') + '.out', 'r', 'utf8') as pt:
                            target_parsed = self.read_parsed(pt)

                    for sentence in range(wmt.sentences(wmt.reference_path(self.dir, dataset.name, lp))):
                        self.plain.append(self.sentence_tuple(dataset=dataset.name,
                                                              lp=lp,
                                                              system=system_name,
                                                              sentence_num=sentence + 1,
                                                              target_sentence=target_sentences[sentence],
                                                              reference_sentence=reference_sentences[sentence]))
                    if parsed:
                        for sentence in range(wmt.sentences(wmt.reference_path(self.dir, dataset.name, lp))):
                            self.parsed.append(self.sentence_tuple(dataset=dataset.name,
                                                                   lp=lp,
                                                                   system=system_name,
                                                                   sentence_num=sentence + 1,
                                                                   target_sentence=target_parsed[sentence],
                                                                   reference_sentence=reference_parsed[sentence]))


            self.datasets.append(dataset)

    def write_meta_data(self, items, path):
        with open(path, 'w') as o:
            for item in items:
                o.write("{}\t{}\t{}\t{}\n".format(item.dataset, item.lp, item.system, item.sentence_num))

    def read_meta_data(self, path):
        data = []
        with open(path) as i:
            lines = i.readlines()
        for line in lines:
            item = line.strip().split('\t')
            data.append(item)
        return data

    def get_plain_data(self):
        return self.plain

    def get_parsed_data(self):
        return self.parsed

    def generate_sample(self, size):
        plain, parsed = zip(*random.sample(list(zip(self.plain, self.parsed)), size))
        return plain, parsed

    def read_parsed(self, FILE):

        lines = FILE.readlines()
        sentences = []
        sentence = []
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                sentences.append(sentence)
            if line.startswith('Sentence #'):
                if i != 0:
                    sentences.append(sentence)
                    sentence = []
                sentence.append(line)
            else:
                sentence.append(line)
        return sentences

    def write_parsed(self, parsed_items):

        path_tgt = usr(self.cfg.get('Data', 'working_dir') + '/' + 'tgt.parse')
        path_ref = usr(self.cfg.get('Data', 'working_dir') + '/' + 'ref.parse')

        with codecs.open(path_tgt, 'w', 'utf8') as o:
            for i, item in enumerate(parsed_items):
                for line in item.target_sentence:
                    if line.startswith('Sentence #'):
                        o.write(wmt.substitute_line_number(line, i + 1))
                    else:
                        o.write(line)

        with codecs.open(path_ref, 'w', 'utf8') as o:
            for i, item in enumerate(parsed_items):
                for line in item.reference_sentence:
                    if line.startswith('Sentence #'):
                        o.write(wmt.substitute_line_number(line, i + 1))
                    else:
                        o.write(line)

    def write_plain(self, plain_items):
        path_tgt = usr(self.cfg.get('Data', 'working_dir') + '/' + 'tgt.txt')
        path_ref = usr(self.cfg.get('Data', 'working_dir') + '/' + 'ref.txt')

        with codecs.open(path_tgt, 'w', 'utf8') as o:
            for item in plain_items:
                o.write(item.target_sentence)

        with codecs.open(path_ref, 'w', 'utf8') as o:
            for item in plain_items:
                o.write(item.reference_sentence)

    def write_dataset(self, parsed=False, verbose=False):

        print("Copying dataset to " + self.cfg.get('Data', 'working_dir') + ' ...')

        path_tgt = usr(self.cfg.get('Data', 'working_dir') + '/' + 'tgt.txt')
        path_ref = usr(self.cfg.get('Data', 'working_dir') + '/' + 'ref.txt')

        counter_tgt = 0
        counter_ref = 0

        with codecs.open(path_tgt, 'w', 'utf8') as output_tgt:
            with codecs.open(path_ref, 'w', 'utf8') as output_ref:

                for dataset in self.datasets:
                    for lp in sorted(dataset.system_names.keys()):

                        with codecs.open(wmt.reference_path(self.dir, dataset.name, lp), 'r', 'utf8') as input_ref:
                            ref_lines = input_ref.readlines()

                        for sys_name in dataset.system_names[lp]:
                            counter_sys = 0
                            with codecs.open(wmt.system_path(self.dir, dataset.name, lp, sys_name), 'r', 'utf8') as input_sys:
                                for line in input_sys.readlines():
                                    counter_tgt += 1
                                    counter_sys += 1
                                    if parsed and line.startswith('Sentence #'):
                                        output_tgt.write(wmt.substitute_line_number(line, counter_tgt))
                                    else:
                                        if verbose:
                                            output_tgt.write('{}\t{}\t{}\t{}\t{}'.format(dataset.name,
                                                                                         lp,
                                                                                         sys_name,
                                                                                         counter_sys,
                                                                                         line))
                                        else:
                                            output_tgt.write(line)

                                for line in ref_lines:
                                    counter_ref += 1
                                    if parsed and line.startswith('Sentence #'):
                                        output_ref.write(wmt.substitute_line_number(line, counter_ref))
                                    else:
                                        output_ref.write(line)

    def write_scores_wmt_format(self, scores, metric='metric', output_path='scores.txt'):

        with open(output_path, 'w') as o:
            counter = 0
            for dataset in self.datasets:
                for lp in sorted(dataset.system_names.keys()):
                    for sys_name in dataset.system_names[lp]:
                        for i in range(dataset.number_sentences[lp]):
                            o.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(metric, dataset.name, lp, sys_name, str(i + 1), scores[counter]))
                            counter += 1

    @staticmethod
    def write_scores_meta(scores, data, metric='metric', output_path='scores.txt'):

        with open(output_path, 'w') as o:
            counter = 0
            for dataset, lp, system, number in data:
                o.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(metric, dataset, lp, system, str(number), scores[counter]))
                counter += 1
