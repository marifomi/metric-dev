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

    def extract_features(self, **kwargs):

        # Receives as optional arguments the names of the files:
        # tgt_plain, ref_plain
        # tgt_token, ref_token
        # tgt_parse, ref_parse

        tools = RunTools()

        if 'tgt_parse' in kwargs.keys():
            tst_parse = read_sentences(codecs.open(kwargs['tgt_parse'], 'r', 'utf-8'))
            ref_parse = read_sentences(codecs.open(kwargs['ref_parse'], 'r', 'utf-8'))

        if 'bleu' in self.names:
            if not os.path.exists(kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.bleu.scores'):
                tools.bleu(kwargs['src_plain'], kwargs['tgt_plain'], kwargs['ref_plain'],
                           kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.bleu.scores')
            bleu = tools.get_bleu(kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.bleu.scores')

        if 'meteor' in self.names:
            if not os.path.exists(kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor.scores'):
                tools.meteor(kwargs['tgt_plain'], kwargs['ref_plain'], 'en',
                             kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor.scores',
                             alignments=kwargs['align_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor')
            meteor = tools.get_meteor(kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor.scores')

        if self.mode == 'adequacy' or self.mode == 'all':
            if kwargs['aligner'] == 'meteor':
                if not os.path.exists(kwargs['align_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor' + '-align.out'):
                    tools.meteor(kwargs['tgt_plain'], kwargs['ref_plain'], 'en',
                                kwargs['baseline_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor.scores',
                                alignments=kwargs['align_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.meteor')
                alignments = tools.get_alignments(kwargs['tgt_plain'], kwargs['align_dir'], kwargs['aligner'])
            else:
                tools.run_aligner_cobalt(kwargs['tgt_parse'], kwargs['ref_parse'], kwargs['align_dir'])
                alignments = tools.get_alignments(kwargs['tgt_parse'], kwargs['align_dir'], kwargs['aligner'])


        if self.mode == 'fluency_sent' or self.mode == 'all':
            tools.run_quest_sent(cfg.get('Resources', 'quest_dir'), cfg.get('Resources', 'quest_config_sent'),
                                 'spanish', 'english', kwargs['ref_plain'], kwargs['tgt_plain'],
                                 cfg.get('Resources', 'quest_output') + '/' + 'sent_output.txt')
            quest_sent_features = tools.get_quest_sent(cfg.get('Resources', 'quest_output') + '/' + 'sent_output.txt',
                                                       cfg.get('Resources', 'quest_features_sent')
                                                       )

        if self.mode == 'fluency_word' or self.mode == 'all':
            if kwargs['aligner'] == 'meteor':
                tools.tokenize_from_aligner(kwargs['align_dir'] + '/' + kwargs['tgt_plain'].split('/')[-1] + '.' +
                                            kwargs['aligner'] + '-align.out',
                                            kwargs['tgt_plain'] + '.align.tok',
                                            kwargs['ref_plain'] + '.align.tok',
                                            )
            else:
                tools.tokenize_from_parse(kwargs['tgt_parse'], kwargs['tgt_plain'] + '.stan.tok')
                tools.tokenize_from_parse(kwargs['ref_parse'], kwargs['ref_plain'] + '.stan.tok')

            tools.run_quest_word(cfg.get('Resources', 'quest_dir'), cfg.get('Resources', 'quest_config_word'),
                                 'spanish', 'english', kwargs['ref_plain'] + '.stan.tok', kwargs['tgt_plain'] + '.stan.tok',
                                 cfg.get('Resources', 'quest_output') + '/' + 'word_output.txt')
            quest_word_features = tools.get_quest_word(kwargs['tgt_plain'] + '.stan.tok',
                                                       cfg.get('Resources', 'quest_output') + '/' + 'word_output.txt',
                                                       cfg.get('Resources', 'quest_features_word')
                                                       )

        # get sent_length
        data_len = self.get_len(kwargs['tgt_plain'])
        for i, sentence in enumerate(range(data_len)):

            my_sentence_tgt = Sentence()
            my_sentence_ref = Sentence()

            phr_feats = []

            if 'tgt_parse' in kwargs.keys():
                candidate_parsed = prepareSentence2(tst_parse[i])
                reference_parsed = prepareSentence2(ref_parse[i])
                my_sentence_tgt.add_parse(candidate_parsed)
                my_sentence_ref.add_parse(reference_parsed)

            # Add tokenization
            if kwargs['aligner'] == 'meteor':
                cand_tok = tools.get_tokens(kwargs['tgt_plain'] + '.align.tok')
            else:
                cand_tok = tools.get_tokens(kwargs['tgt_plain'] + '.stan.tok')
            my_sentence_tgt.add_tokenized(cand_tok[i])


            if self.mode == 'adequacy' or self.mode == 'all':
                my_sentence_tgt.add_alignments(alignments[i])
                my_sentence_ref.add_alignments(alignments[i])

            if self.mode == 'fluency_sent' or self.mode == 'all':
                my_sentence_tgt.add_quest_sent(quest_sent_features[i])

            if self.mode == 'fluency_word' or self.mode == 'all':
                my_sentence_tgt.add_quest_word(quest_word_features[i])

            if 'bleu' in self.names:
                my_sentence_tgt.add_bleu(bleu[i])

            if 'meteor' in self.names:
                my_sentence_tgt.add_meteor(meteor[i])

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

    @staticmethod
    def get_len(my_file):
        return sum(1 for line in open(my_file))


def main():

    tst = os.getcwd() + '/' + 'data' + '/' + 'system'
    ref = os.getcwd() + '/' + 'data' + '/' + 'reference'
    output = open(os.getcwd() + '/' + 'test' + '/' + 'features.tsv', 'w')

    extractor = FeatureExtractor()
    extractor.read_config()
    extractor.get_feature_names()
    extractor.extract_features(tgt_plain=tst,
                               ref_plain=ref,
                               tgt_parse=tst + '.parse',
                               ref_parse=ref + '.parse',
                               align_dir=os.getcwd() + '/' + 'alignments')

    for phr in extractor.vals:
        output.write('\t'.join([str(x) for x in phr]) + '\n')

    output.close()

if __name__ == '__main__':
    main()