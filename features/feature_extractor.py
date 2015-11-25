__author__ = 'MarinaFomicheva'

from utils import load_resources
from utils.core_nlp_utils import read_sentences
from ConfigParser import ConfigParser
from alignment.aligner import Aligner
from features.utils.lang_model_generator import LangModelGenerator

class FeatureExtractor(object):

    config = ConfigParser()

    def __int__(self):
        self.features = []
        self.config.readfp(open('config/metric.cfg'))

    def get_feature_names(self, feature_names_file_name):

        features_names_file = open(feature_names_file_name, 'r')
        for line in features_names_file:
            self.features.append(line.strip())

    def read_config(self):

        self.mode = self.config.get('Features', 'mode')

    def extract_features(self, tst_file_name, ref_file_name):

        tst_file = open(tst_file_name, 'r')
        ref_file = open(ref_file_name, 'r')
        tst_phrases = read_sentences(tst_file)
        ref_phrases = read_sentences(ref_file)

        if self.mode == 'adequacy' or self.mode == 'all':

            ppdb_file_name = self.config.get('Paths', 'ppdb_file_name')
            vectors_file_name = self.config.get('Paths', 'vectors_file_name')

            load_resources.load_ppdb(ppdb_file_name)
            load_resources.load_word_vectors(vectors_file_name)
            aligner = Aligner('english')

        if self.mode == 'fluency' or self.mode == 'all':

            lm = LangModelGenerator()
            lm.tokenize(tst_file_name)
            lm.produce_ppl(tst_file_name, tst_file_name + '.ppl', self.config.get('Resources', 'lm'), 3)
            lm_features = lm.read_ppl(tst_file_name + '.ppl')









            alignments = aligner.align(test_data[i], phrase_ref)
            candidate_parsed = prepareSentence2(test_data[i])
            reference_parsed = prepareSentence2(phrase_ref)







