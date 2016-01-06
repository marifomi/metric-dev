__author__ = 'MarinaFomicheva'

from ConfigParser import ConfigParser
import inspect
import os
import codecs

from src.utils.sentence import Sentence
from src.utils.core_nlp_utils import read_sentences
from src.utils.core_nlp_utils import prepareSentence2
from src.alignment.aligner import Aligner
from src.alignment.alignments_reader import AlignmentsReader
from src.tools.lang_model_generator import LangModelGenerator
from src.features.impl import features
from src.tools.run_tools import RunTools


config = ConfigParser()
config.readfp(open(os.getcwd() + '/config/metric.cfg'))

class FeatureExtractor(object):

    def __init__(self):
        self.vals = []
        self.names = []
        self.mode = str

    def get_feature_names(self):

        file_ = open(config.get('Features', 'path'), 'r')
        for line in file_:
            self.names.append(line.strip())

    def read_config(self):

        self.mode = config.get('Features', 'mode')

    def extract_features(self, tst_file_name, ref_file_name, **kwargs):

        tst_parse = open(tst_file_name + '.parse', 'r')
        ref_parse = open(ref_file_name + '.parse', 'r')
        tst_parse_ = read_sentences(tst_parse)
        ref_parse_ = read_sentences(ref_parse)

        tools = RunTools()
        if self.mode == 'adequacy' or self.mode == 'all':
            tools.run_aligner(tst_file_name + '.parse', ref_file_name + '.parse', kwargs['align_dir'])
            alignments = tools.get_alignments(tst_file_name + '.parse', kwargs['align_dir'])

        if self.mode == 'fluency' or self.mode == 'all':
            quest_dir = config.get('Resources', 'quest_dir')
            quest_config = config.get('Resources', 'quest_config')
            quest_out = config.get('Resources', 'quest_output')
            src_tok = ref_file_name + '.tok'
            tgt_tok = tst_file_name + '.tok'
            out_path = quest_out + '/' + 'output.txt'
            tools.run_quest_word(quest_dir, quest_config, 'spanish', 'english', src_tok, tgt_tok, out_path)
            quest_word_features = tools.get_quest_word(tgt_tok, out_path)

        for i, sentence in enumerate(tst_parse_):

            phr_feats = []

            candidate_parsed = prepareSentence2(sentence)
            reference_parsed = prepareSentence2(ref_parse_[i])

            my_sentence_tgt = Sentence()
            my_sentence_ref = Sentence()

            my_sentence_tgt.add_parse(candidate_parsed)
            my_sentence_ref.add_parse(reference_parsed)

            if self.mode == 'adequacy' or self.mode == 'all':
                my_sentence_tgt.add_alignments(alignments[i])
                my_sentence_ref.add_alignments(alignments[i])

            if self.mode == 'fluency' or self.mode == 'all':
                my_sentence_tgt.add_quest_word(quest_word_features[i])

            for name, my_class in sorted(inspect.getmembers(features)):

                if 'Abstract' in name or 'Scorer' in name:
                    continue

                if not inspect.isclass(my_class):
                    continue

                instance = my_class()

                if instance.get_name() not in self.names:
                    continue

                instance.run(my_sentence_tgt, my_sentence_ref)
                phr_feats.append(instance.get_value())

            self.vals.append(phr_feats)


def main():

    tst = os.getcwd() + '/' + 'data' + '/' + 'system'
    ref = os.getcwd() + '/' + 'data' + '/' + 'reference'
    output = open(os.getcwd() + '/' + 'output' + '/' + 'features.tsv', 'w')

    extractor = FeatureExtractor()
    extractor.read_config()
    extractor.get_feature_names()
    extractor.extract_features(tst, ref, align_dir=os.getcwd() + '/' + 'alignments')
    for phr in extractor.vals:
        output.write('\t'.join([str(x) for x in phr]) + '\n')

    output.close()

if __name__ == '__main__':
    main()