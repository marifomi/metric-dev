__author__ = 'MarinaFomicheva'

import os
from src.tools.run_tools import RunTools
from src.features.feature_extractor import FeatureExtractor

class WMT(object):

    def __init__(self, dir_, dataset, lps):
        self.dir = dir_
        self.dataset = dataset
        self.lps = lps

    def process_wmt(self):

        features = open(os.getcwd() + '/' + 'output' + '/' + self.dataset + '.' + 'features.tsv', 'w')
        meta_data = open(os.getcwd() + '/' + 'output' + '/' + self.dataset + '.' + 'meta_data.tsv', 'w')

        for lp in sorted(os.listdir(self.dir + '/' + 'parsed' + '/' + 'system-outputs')):
            if not lp == 'cs-en':
                continue
            ref_file = self.dir + '/' + 'parsed' + '/' + 'references' + '/' + self.dataset + '-ref.' + lp + '.out'
            for sys in sorted(os.listdir(self.dir + '/' + 'parsed' + '/' + 'system-outputs' + '/' + lp)):
                print sys
                sys_file = self.dir + '/' + 'parsed' + '/' + 'system-outputs' + '/' + lp + '/' + sys

                extractor = FeatureExtractor()
                extractor.read_config()
                extractor.get_feature_names()
                extractor.extract_features(sys_file, ref_file, align_dir=os.getcwd() + '/' + 'alignments')

                for phr in extractor.vals:
                    meta_data.write('\t'.join([lp, get_sys_name(sys, self.dataset)]) + '\n')
                    features.write('\t'.join([str(x) for x in phr]) + '\n')

        features.close()
        meta_data.close()


def main():

    data_dir = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt15-data')
    dataset = 'newstest2015'
    lps = ['cs-en']
    wmt = WMT(data_dir, dataset, lps)
    wmt.process_wmt()

def get_sys_name(sys_id, dataset):
    sys_id = sys_id.replace('.txt', '')
    sys_id = sys_id.replace('.out', '')
    sys_id = sys_id.replace(dataset, '')
    sys_name = '.'.join(sys_id.split('.')[1:-1])
    return sys_name

if __name__ == '__main__':
    main()