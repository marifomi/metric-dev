import codecs

from os.path import expanduser as usr
from os.path import exists
from utils import wmt
from collections import defaultdict
from json import loads


class Dataset(object):

    def __init__(self, name):
        self.name = name
        self.system_names = defaultdict(list)
        self.number_sentences = {}


class RankingData(object):

    def __init__(self, config):
        self.cfg = config
        self.dir = self.cfg.get('Paths', 'input_dir')
        self.datasets = []

    def read_dataset(self):

        dataset_names = wmt.get_datasets(self.dir)

        for dataset_name in dataset_names:
            dataset = Dataset(dataset_name)
            lang_pairs = wmt.get_lang_pairs(self.dir, dataset_name)

            for lp in lang_pairs:
                if len(loads(self.cfg.get('Settings', 'lang_pairs'))) > 0 and lp not in loads(self.cfg.get('Settings', 'lang_pairs')):
                    continue

                system_names = wmt.get_system_names(self.dir, dataset_name, lp)
                number_sentences = wmt.sentences(wmt.reference_path(self.dir, dataset.name, lp))

                dataset.system_names[lp] = system_names
                dataset.number_sentences[lp] = number_sentences

            self.datasets.append(dataset)

    def write_dataset(self):

        print("Copying dataset to " + self.cfg.get('Paths', 'working_dir') + ' ...')

        path_tgt = usr(self.cfg.get('Paths', 'working_dir') + '/' + 'tgt.txt')
        path_ref = usr(self.cfg.get('Paths', 'working_dir') + '/' + 'ref.txt')

        with codecs.open(path_tgt, 'w', 'utf8') as output_tgt:
            with codecs.open(path_ref, 'w', 'utf8') as output_ref:

                for dataset in self.datasets:
                    for lp in sorted(dataset.system_names.keys()):

                        with codecs.open(wmt.reference_path(self.dir, dataset.name, lp), 'r', 'utf8') as input_ref:
                            ref_lines = input_ref.readlines()

                        for sys_name in dataset.system_names[lp]:
                            with codecs.open(wmt.system_path(self.dir, dataset.name, lp, sys_name), 'r', 'utf8') as input_sys:
                                for line in input_sys.readlines():
                                    output_tgt.write(line)

                                for line in ref_lines:
                                    output_ref.write(line)
