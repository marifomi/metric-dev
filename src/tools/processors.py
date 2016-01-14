__author__ = 'MarinaFomicheva'

from src.tools.abstract_processor import AbstractProcessor
from src.alignment.aligner import Aligner
from src.utils.cobalt_align_reader import CobaltAlignReader
from src.utils.meteor_align_reader import MeteorAlignReader
import codecs
import os
import subprocess
import shutil
import numpy as np
from src.utils.core_nlp_utils import read_sentences
from src.utils.core_nlp_utils import prepareSentence2
import re
from src.utils.features_reader import FeaturesReader
import src.utils.txt_xml as xml


class Bleu(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'bleu')

    def run(self, config):

        src_path = config.get('Data', 'src')
        tgt_path = config.get('Data', 'tgt')
        ref_path = config.get('Data', 'ref')

        if not os.path.exists(src_path + '.xml'):
            xml.run(src_path, tgt_path, ref_path)

        bleu_path = config.get('Metrics', 'bleu')
        my_file = config.get('Metrics', 'dir') + '/' + 'bleu.scores'
        o = open(my_file, 'w')
        subprocess.call(['perl', bleu_path, '-b', '-d', str(2), '-r', ref_path + '.xml',
                         '-t', tgt_path + '.xml',
                         '-s', src_path + '.xml'], stdout=o)
        o.close()

    def get(self, config):

        result = []
        scores_file = config.get('Metrics', 'dir') + '/' + 'bleu.scores'
        for line in open(scores_file).readlines():
            if not line.startswith('  BLEU'):
                continue
            result.append(float(re.sub(r'^.+= ([01]\.[0-9]+).+$', r'\1', line.strip())))

        AbstractProcessor.set_result(self, result)


class MeteorScorer(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'meteor_scorer')

    def run(self, config):

        tgt_path = config.get('Data', 'tgt')
        ref_path = config.get('Data', 'ref')

        meteor = config.get('Metrics', 'meteor')
        lang = config.get('Setting', 'tgt_lang')
        my_file = config.get('Metrics', 'dir') + '/' + 'meteor.scores'

        o = open(my_file, 'w')
        subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang, '-norm'], stdout=o)
        o.close()

    def get(self, config):

        result = []
        scores_file = config.get('Metrics', 'dir') + '/' + 'meteor.scores'

        for line in open(scores_file).readlines():
            if not line.startswith('Segment '):
                continue
            result.append(float(line.strip().split('\t')[1]))

        AbstractProcessor.set_result(self, result)


class MeteorAligner(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'meteor_aligner')

    def run(self, config):

        tgt_path = config.get('Data', 'tgt')
        ref_path = config.get('Data', 'ref')

        meteor = config.get('Metrics', 'meteor')
        lang = config.get('Setting', 'tgt_lang')

        tgt_file_name = tgt_path.split('/')[-1]
        prefix = config.get('Alignment', 'dir') + '/' + tgt_file_name + '.' + config.get('Alignment', 'aligner')
        subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang, '-norm', '-writeAlignments', '-f', prefix])

    def get(self, config):

        tgt_path = config.get('Data', 'tgt')
        align_dir = config.get('Alignments', 'dir')
        reader = MeteorAlignReader()

        result = reader.read(align_dir + '/' + tgt_path.split('/')[-1] + '.' + 'meteor' + '-align.out')
        AbstractProcessor.set_result(self, result)


class CobaltAligner(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'cobalt_aligner')

    def run(self, config):

        tgt_path = config.get('Data', 'tgt') + '.parse'
        ref_path = config.get('Data', 'ref') + '.parse'
        align_dir = config.get('Alignments', 'dir')

        if os.path.exists(align_dir + '/' + tgt_path.split('/')[-1] + '.cobalt-align.out'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        aligner = Aligner('english')
        aligner.align_documents(tgt_path, ref_path)
        aligner.write_alignments(align_dir + '/' + tgt_path.split('/')[-1] + '.cobalt-align.out')

    def get(self, config):

        tgt_path = config.get('Data', 'tgt') + '.parse'
        align_dir = config.get('Alignments', 'dir')
        reader = CobaltAlignReader()

        result = reader.read(align_dir + '/' + tgt_path.split('/')[-1] + '.' + 'cobalt' + '-align.out')
        AbstractProcessor.set_result(self, result)


class Tokenizer(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'tokenizer')

    def run(self, config):

        method = config.get('Tokenizer', 'method')

        if method == 'aligner':
            aligner = config.get('Alignment', 'aligner')
            if aligner == 'meteor':
                self.tokenize_from_aligner(config)
            elif aligner == 'cobalt':
                self.tokenize_from_parse(config)
            else:
                print "Aligner is not defined"
        elif method == 'quest':
            self.tokenize_quest(config)
        else:
            print "Tokenizer is not defined"

        result = self.get(config)

        AbstractProcessor.set_result(self, result)

    def tokenize_from_aligner(self, config):

        align_dir = config.get('Alignment', 'dir')
        aligner = 'meteor'
        tgt_path = config.get('Data', 'tgt')
        ref_path = config.get('Data', 'ref')

        input_file = align_dir + '/' + tgt_path.split('/')[-1] + '.' + aligner + '-align.out'
        output_file_tgt = tgt_path + '.token'
        output_file_ref = ref_path + '.token'

        if os.path.exists(output_file_tgt):
            print("The file " + output_file_tgt + " is already tokenized.\n Tokenizer will not run.")
            return

        o_tgt = codecs.open(output_file_tgt, 'w', 'utf-8')
        o_ref = codecs.open(output_file_ref, 'w', 'utf-8')
        lines = codecs.open(input_file, 'r', 'utf-8').readlines()

        for i, line in enumerate(lines):
            if line.startswith('Line2Start:Length'):
                continue

            if line.startswith('Alignment\t'):
                words_test = lines[i + 1].strip().split(' ')
                words_ref = lines[i + 2].strip().split(' ')
                o_tgt.write(' '.join(words_test) + '\n')
                o_ref.write(' '.join(words_ref) + '\n')

        o_tgt.close()
        o_ref.close()

    def tokenize_from_parse(self, config):

        input_file = config.get('Data', 'tgt') + '.parse'
        output_file = config.get('Data', 'tgt') + '.token'
        if os.path.exists(output_file):
            print("The file " + output_file + " is already tokenized.\n Tokenizer will not run.")
            return

        o = codecs.open(output_file, 'w', 'utf-8')
        lines = codecs.open(input_file, 'r', 'utf-8')
        processed = read_sentences(lines)

        for i, sentence in enumerate(processed):
            parsed = prepareSentence2(sentence)
            words = []

            for item in parsed:
                word = re.sub("``|''", "\"", item.form)
                word = re.sub("-LRB-", "(", word)
                word = re.sub("-RRB-", ")", word)
                word = re.sub("-LSB-", "[", word)
                word = re.sub("-RSB-", "]", word)
                word = re.sub("-LCB-", "{", word)
                word = re.sub("-RCB-", "}", word)
                words.append(word)

            o.write(' '.join(words) + '\n')

        o.close()

    def tokenize_quest(self, config):

        # Using quest tokenizer

        tokenizer = config.get('Tokenizer', 'tool_path')
        input_file = config.get('Data', 'tgt')
        myinput = open(input_file, 'r')

        myoutput = open(input_file + '.token', 'w')
        p = subprocess.Popen(['perl', tokenizer], stdin=myinput, stdout=myoutput)
        p.wait()
        myoutput.flush()

        print "Tokenization finished!"

    def get(self, config):

        sents_tokens = []

        input_file = config.get('Data', 'tgt') + '.token'
        lines = codecs.open(input_file, 'r', 'utf-8')
        for line in lines:
            tokens = line.strip().split(' ')
            sents_tokens.append(tokens)

        lines.close()
        return sents_tokens


class QuestSentence(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'quest_sentence')

    def run(self, config):

        # Plain files, no tokenization!
        # Check quest configuration file!

        quest_dir = config.get('Quest', 'path')
        src_lang = config.get('Settings', 'src_lang')
        tgt_lang = config.get('Settings', 'tgt_lang')
        src_path = config.get('Data', 'src')
        tgt_path = config.get('Data', 'tgt')
        quest_config = config.get('Quest', 'config') + '/' + 'config.' + 'sl.' + tgt_lang + '.properties'
        out_file = config.get('Quest', 'output') + 'quest.sl.out'

        subprocess.call(['java',
                         '-cp',
                         quest_dir + '/' + 'dist/Quest__.jar',
                         'shef.mt.SentenceLevelFeatureExtractor',
                         '-lang',
                         src_lang,
                         tgt_lang,
                         '-input',
                         src_path,
                         tgt_path,
                         '-config',
                         quest_config,
                         '-tok',
                         '-output_file',
                         out_file
        ])
        shutil.rmtree(os.getcwd() + '/' + 'input')

    def get(self, config):

        # Make sure feature file is the same as stated in quest config file
        # Later change to read features from config file

        features_file = config.get('Quest', 'features_sent')
        output_quest = config.get('Quest', 'output') + 'quest.sl.out'
        reader = FeaturesReader()
        features = reader.read_features(features_file)

        sentences = open(output_quest, 'r').readlines()
        result = []

        for sent in sentences:
            feats = {}

            for i, val in enumerate(sent.strip().split('\t')):
                feats[features[i]] = float(val)

            result.append(feats)

        AbstractProcessor.set_result(self, result)


class QuestWord(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'quest_word')

    def run(self, config):

        # Tokenized input files !
        # Check quest configuration file!

        quest_dir = config.get('Quest', 'path')
        src_lang = config.get('Settings', 'src_lang')
        tgt_lang = config.get('Settings', 'tgt_lang')
        src_path = config.get('Data', 'src') + '.token'
        tgt_path = config.get('Data', 'tgt') + '.token'
        quest_config = config.get('Quest', 'config') + '/' + 'config.' + 'wl.' + tgt_lang + '.properties'
        out_file = config.get('Quest', 'output') + 'quest.wl.out'

        subprocess.call(['java',
                         '-cp',
                         quest_dir + '/' + 'dist/Quest__.jar',
                         'shef.mt.WordLevelFeatureExtractor',
                         '-lang',
                         src_lang,
                         tgt_lang,
                         '-input',
                         src_path,
                         tgt_path,
                         '-config',
                         quest_config,
                         '-mode',
                         'all',
                         '-output_file',
                         out_file
        ])
        shutil.rmtree(os.getcwd() + '/' + 'input')

    def get(self, config):

        # Make sure feature file is the same as stated in quest config file
        # Later change to read features from config file

        features_file = config.get('Quest', 'features_word')
        output_quest = config.get('Quest', 'output') + 'quest.wl.out'
        input_token = config.get('Data', 'tgt') + '.token'

        reader = FeaturesReader()
        features = reader.read_features(features_file)

        lengths = self.sent_length(input_token)
        words = open(output_quest, 'r').readlines()
        result = []

        cnt_words = 0
        cnt_sentences = 0
        sent_words = []
        for word in words:

            word_feats = {}
            for i, val in enumerate(word.strip().split('\t')):
                word_feats[features[i]] = int(val.split('=')[1])

            if cnt_words >= np.sum(lengths[:cnt_sentences + 1]):
                result.append(sent_words)
                sent_words = []
                cnt_sentences += 1
            cnt_words +=1
            sent_words.append(word_feats)

        result.append(sent_words)

        AbstractProcessor.set_result(self, result)

    @staticmethod
    def sent_length(file_):

        lengths = []
        sents = open(file_, 'r').readlines()
        for sent in sents:
            lengths.append(len(sent.split(' ')))

        return lengths