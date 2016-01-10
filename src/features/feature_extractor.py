__author__ = 'MarinaFomicheva'

from ConfigParser import ConfigParser
import inspect
import os
import codecs

from src.utils.sentence import Sentence
from src.utils.core_nlp_utils import read_sentences
from src.utils.core_nlp_utils import prepareSentence2
from src.features.impl import features
from src.tools.run_tools import RunTools


cfg = ConfigParser()
cfg.readfp(open(os.getcwd() + '/config/metric.cfg'))

class FeatureExtractor(object):

    def __init__(self):
        self.vals = []
        self.names = []
        self.mode = str

    def get_feature_names(self):

        file_ = open(cfg.get('Features', 'path'), 'r')
        for line in file_:
            self.names.append(line.strip())

    def read_config(self):

        self.mode = cfg.get('Features', 'mode')

    def extract_features(self, tst_file_name, ref_file_name, **kwargs):

        tst_parse = read_sentences(codecs.open(tst_file_name + '.parse', 'r', 'utf-8'))
        ref_parse = read_sentences(codecs.open(ref_file_name + '.parse', 'r', 'utf-8'))

        tools = RunTools()
        if self.mode == 'adequacy' or self.mode == 'all':
            tools.run_aligner(tst_file_name + '.parse', ref_file_name + '.parse', kwargs['align_dir'])
            alignments = tools.get_alignments(tst_file_name + '.parse', kwargs['align_dir'])

        if self.mode == 'fluency_sent' or self.mode == 'all':
            tools.run_quest_sent(cfg.get('Resources', 'quest_dir'), cfg.get('Resources', 'quest_config_sent'),
                                 'spanish', 'english', ref_file_name, tst_file_name,
                                 cfg.get('Resources', 'quest_output') + '/' + 'sent_output.txt')
            quest_sent_features = tools.get_quest_sent(cfg.get('Resources', 'quest_output') + '/' + 'sent_output.txt',
                                                       cfg.get('Resources', 'quest_features_sent')
                                                       )

        if self.mode == 'fluency_word' or self.mode == 'all':
            tools.tokenize_from_parse(tst_file_name + '.parse', tst_file_name + '.stan.tok')
            tools.tokenize_from_parse(ref_file_name + '.parse', ref_file_name + '.stan.tok')
            tools.run_quest_word(cfg.get('Resources', 'quest_dir'), cfg.get('Resources', 'quest_config_word'),
                                 'spanish', 'english', ref_file_name + '.stan.tok', tst_file_name + '.stan.tok',
                                 cfg.get('Resources', 'quest_output') + '/' + 'word_output.txt')
            quest_word_features = tools.get_quest_word(tst_file_name + '.stan.tok',
                                                       cfg.get('Resources', 'quest_output') + '/' + 'word_output.txt',
                                                       cfg.get('Resources', 'quest_features_word')
                                                       )

        for i, sentence in enumerate(tst_parse):

            phr_feats = []

            candidate_parsed = prepareSentence2(sentence)
            reference_parsed = prepareSentence2(ref_parse[i])

            my_sentence_tgt = Sentence()
            my_sentence_ref = Sentence()

            my_sentence_tgt.add_parse(candidate_parsed)
            my_sentence_ref.add_parse(reference_parsed)

            if self.mode == 'adequacy' or self.mode == 'all':
                my_sentence_tgt.add_alignments(alignments[i])
                my_sentence_ref.add_alignments(alignments[i])

            if self.mode == 'fluency_sent' or self.mode == 'all':
                my_sentence_tgt.add_quest_sent(quest_sent_features[i])

            if self.mode == 'fluency_word' or self.mode == 'all':
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
    output = open(os.getcwd() + '/' + 'test' + '/' + 'test.tsv', 'w')

    extractor = FeatureExtractor()
    extractor.read_config()
    extractor.get_feature_names()
    extractor.extract_features(tst, ref, align_dir=os.getcwd() + '/' + 'alignments')
    for phr in extractor.vals:
        output.write('\t'.join([str(x) for x in phr]) + '\n')

    output.close()

if __name__ == '__main__':
    main()