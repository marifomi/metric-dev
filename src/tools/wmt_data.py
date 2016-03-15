__author__ = 'MarinaFomicheva'

from collections import defaultdict
import os
import codecs
from json import loads
from src.utils.core_nlp_utils import read_parsed_sentences
import re

class Translations(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

class WmtData(object):

    def __init__(self):

        self.features = []
        self.lp_sizes = {}
        self.lp_systems = defaultdict(list)

    def add_features(self, features):
        self.features = features

    def get_lp_sizes(self):
        return self.lp_sizes

    def get_lp_systems(self):
        return self.lp_systems

    @staticmethod
    def wmt_format(config, scores, lp_sizes, lp_systems, sample='.test'):

        o = open(config.get('WMT', 'wmt_output'), 'w')

        for lp in sorted(lp_systems.keys()):

            for i, sys in enumerate(sorted(lp_systems[lp])):

                for k in range(lp_sizes[lp]):

                    idx = lp_sizes[lp] * i + k
                    dataset = config.get('WMT', 'dataset' + '_' + sample.replace('.', ''))
                    phr = str(k + 1)
                    score = str(scores[idx])

                    print >>o, config.get('WMT', 'wmt_name') + '\t' + dataset + '\t' + lp + '\t' + sys + '\t' + phr +\
                            '\t' + str(score)

        o.close()

    def preprocess(self, config, sample, data_type):

        if data_type == 'parse':
            dtype_name = 'parse'
        else:
            dtype_name = ''

        o_ref = codecs.open(config.get('WMT', 'ref') + '.' + dtype_name + sample, 'w', 'utf-8')
        o_tgt = codecs.open(config.get('WMT', 'tgt') + '.' + dtype_name + sample, 'w', 'utf-8')

        wmt_dir = config.get('WMT', sample.replace('.', ''))
        dataset = config.get('WMT', 'dataset' + '_' + sample.replace('.', ''))

        self.features = []
        self.lp_sizes = {}
        self.lp_systems = defaultdict(list)

        for lp in sorted(os.listdir(wmt_dir + '/' + data_type + '/' + 'system-outputs' + '/' + dataset)):

            if lp.startswith('.DS'):
                continue

            if '-en' not in lp:
                continue

            if not config.get('WMT', 'directions') == 'None' and lp not in loads(config.get('WMT', 'directions')):
                continue

            fref = self.set_ref_file(wmt_dir, data_type, dataset, lp)
            sents_ref = self.get_sentences(fref, data_type)

            count_sys = 0
            for sys in sorted(os.listdir(wmt_dir + '/' + data_type + '/' + 'system-outputs' + '/' + dataset + '/' + lp)):
                if sys.startswith('.'):
                    continue

                ftgt = wmt_dir + '/' + data_type + '/' + 'system-outputs' + '/' + dataset + '/' + lp + '/' + sys
                sys_name = self.get_sys_name(sys, dataset)
                sents_sys= self.get_sentences(ftgt, data_type)

                self.lp_systems[lp].append(sys_name)

                if not lp in self.lp_sizes.keys():
                    self.lp_sizes[lp] = len(sents_sys)

                for i, sent in enumerate(sents_sys):
                    if data_type == 'parse':
                        idx = self.lp_sizes[lp] * count_sys + i
                        m = re.match(r'Sentence #[0-9]+ ((.|\n)+)$', sent)
                        nsent = 'Sentence #' + str(idx + 1) + ' ' + m.group(1)
                        o_tgt.write(nsent + '\n')
                    else:
                        o_tgt.write(sent + '\n')

                for i, sent in enumerate(sents_ref):

                    if data_type == 'parse':
                        idx = self.lp_sizes[lp] * count_sys + i
                        m = re.match(r'Sentence #[0-9]+ ((.|\n)+)$', sent)
                        nsent = 'Sentence #' + str(idx + 1) + ' ' + m.group(1)
                        o_ref.write(nsent + '\n')
                    else:
                        o_ref.write(sent + '\n')

                count_sys += 1

        o_ref.close()
        o_tgt.close()

    @staticmethod
    def get_sentences(fname, data_type):
        f = codecs.open(fname, 'r', 'utf-8')
        if data_type == 'plain' or data_type == 'token':
            sents = [line.strip() for line in f.readlines()]
            f.close()
            return sents
        elif data_type == 'parse':
            sents = read_parsed_sentences(f)
            f.close()
            return sents
        else:
            print "Unknown data type!"
            return

    @staticmethod
    def set_ref_file(dir_, info_type, dataset, lp):

        if dataset == 'newstest2014':
            if info_type == 'parse':
                return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-ref.' + lp + '.out'
            elif info_type == 'plain':
                return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-ref.' + lp
            elif info_type == 'token':
                return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-ref.' + lp + '.token'
        elif dataset == 'newstest2015':
            if info_type == 'parse':
                return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-' + lp.split('-')[0] + lp.split('-')[1] +\
                '-ref.' + lp.split('-')[1] + '.out'
            elif info_type == 'plain':
                return dir_ + '/' + info_type + '/' + 'references' + '/' + dataset + '/' + dataset + '-' + lp.split('-')[0] + lp.split('-')[1] +\
                '-ref.' + lp.split('-')[1]

    @staticmethod
    def get_sys_name(sys_id, dataset):
        sys_id = sys_id.replace('.txt', '')
        sys_id = sys_id.replace('.out', '')
        sys_id = sys_id.replace('.stan.tok', '')
        sys_id = sys_id.replace(dataset, '')
        sys_name = '.'.join(sys_id.split('.')[1:-1])
        return sys_name

def main():

    from ConfigParser import ConfigParser
    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/system.cfg'))
    wmt = WmtData()
    wmt.preprocess(cfg, 'train', 'parse')

if __name__ == '__main__':
    main()