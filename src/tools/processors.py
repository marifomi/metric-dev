__author__ = 'MarinaFomicheva'

from src.tools.abstract_processor import AbstractProcessor
from src.alignment.aligner import Aligner
from src.alignment.aligner import stopwords
from src.alignment.aligner import punctuations
from src.utils.cobalt_align_reader import CobaltAlignReader
from src.utils.meteor_align_reader import MeteorAlignReader
import codecs
import os
import subprocess
import shutil
import numpy as np
from src.utils.core_nlp_utils import read_parsed_sentences
from src.utils.core_nlp_utils import prepareSentence2
import re
from src.utils.features_reader import FeaturesReader
import src.utils.txt_xml as xml
from gensim.models.word2vec import Word2Vec
from numpy import array
from numpy import zeros


class SentVector(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'sent_vector')

    def run(self, config, sample):
        print "Loading word vectors"

    def get(self, config, sample):

        print "Getting sentence vectors"

        lines_ref = codecs.open(config.get('Data', 'ref') + '.' + 'token' + '.' + sample, 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(config.get('Data', 'tgt') + '.' + 'token' + '.' + sample, 'r', 'utf-8').readlines()

        fvectors = config.get('Vectors', 'path')
        wv = Word2Vec.load_word2vec_format(fvectors, binary=False)

        AbstractProcessor.set_result_tgt(self, self.sents2vec(lines_tgt, wv))
        AbstractProcessor.set_result_ref(self, self.sents2vec(lines_ref, wv))

        wv = None
        print "Finished getting sentence vectors"


    @staticmethod
    def sents2vec(sents, model):

        result = []
        for line in sents:

            tokens = line.strip().split(' ')
            vecs = []

            for token in tokens:
                #if token in punctuations or token in stopwords:
                if token in punctuations:
                    continue
                if token in model.index2word:
                    vecs.append(model[token])

            if len(vecs) == 0:
                result.append(zeros(model.vector_size))
            else:
                result.append(array(vecs).mean(axis=0))

        return result


class Parse(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'parse')

    def run(self, config, sample):
        print "Parse already exist!"

    def get(self, config, sample):

        result_tgt = read_parsed_sentences(codecs.open(config.get('Data', 'tgt') + '.' + 'parse' + '.' + sample, 'r', 'utf-8'))
        result_ref = read_parsed_sentences(codecs.open(config.get('Data', 'ref') + '.' + 'parse' + '.' + sample, 'r', 'utf-8'))

        AbstractProcessor.set_result_tgt(self, result_tgt)
        AbstractProcessor.set_result_ref(self, result_ref)


class Bleu(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'bleu')

    def run(self, config, sample):

        src_path = config.get('Data', 'src') + '.' + sample
        tgt_path = config.get('Data', 'tgt') + '.' + sample
        ref_path = config.get('Data', 'ref') + '.' + sample

        if os.path.exists(config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.bleu.scores'):
            print "Bleu scores already exist!"
            return

        if not os.path.exists(src_path + '.xml'):
            xml.run(src_path, tgt_path, ref_path)

        bleu_path = config.get('Metrics', 'bleu')
        my_file = config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.bleu.scores'
        o = open(my_file, 'w')
        subprocess.call(['perl', bleu_path, '-b', '-d', str(2), '-r', ref_path + '.xml',
                         '-t', tgt_path + '.xml',
                         '-s', src_path + '.xml'], stdout=o)
        o.close()

    def get(self, config, sample):

        result = []
        tgt_path = config.get('Data', 'tgt') + '.' + sample
        scores_file = config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.bleu.scores'
        for line in open(scores_file).readlines():
            if not line.startswith('  BLEU'):
                continue
            result.append(float(re.sub(r'^.+= ([01]\.[0-9]+).+$', r'\1', line.strip())))

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class MeteorScorer(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'meteor')

    def run(self, config, sample):

        tgt_path = config.get('Data', 'tgt') + '.' + sample
        ref_path = config.get('Data', 'ref') + '.' + sample

        if os.path.exists(config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.meteor.scores'):
            print "Meteor scores already exist!"
            return

        meteor = config.get('Metrics', 'meteor')
        lang = config.get('Settings', 'tgt_lang')
        my_file = config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.meteor.scores'

        o = open(my_file, 'w')
        subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang, '-norm'], stdout=o)
        o.close()

    def get(self, config, sample):

        result = []
        tgt_path = config.get('Data', 'tgt') + '.' + sample
        scores_file = config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.meteor.scores'

        for line in open(scores_file).readlines():
            if not line.startswith('Segment '):
                continue
            result.append(float(line.strip().split('\t')[1]))

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class MeteorAligner(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'meteor_aligner')

    def run(self, config, sample):

        tgt_path = config.get('Data', 'tgt') + '.' + sample
        ref_path = config.get('Data', 'ref') + '.' + sample
        tgt_file_name = tgt_path.split('/')[-1]
        name = ''
        if len(config.get('Alignment', 'name')) > 0:
            name = '_' + config.get('Alignment', 'name')

        prefix = config.get('Alignment', 'dir') + '/' + tgt_file_name + '.' + config.get('Alignment', 'aligner') + name

        if os.path.exists(prefix + '-align.out'):
            print "Meteor alignments already exist!"
            return

        meteor = config.get('Metrics', 'meteor')
        lang = config.get('Settings', 'tgt_lang')

        subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang,
                         '-norm', '-writeAlignments', '-f', prefix])

    def get(self, config, sample):

        tgt_path = config.get('Data', 'tgt') + '.' + sample
        align_dir = config.get('Alignment', 'dir')
        aligner = config.get('Alignment', 'aligner')
        name = ''
        if len(config.get('Alignment', 'name')) > 0:
            name = '_' + config.get('Alignment', 'name')
        reader = MeteorAlignReader()

        result = reader.read(align_dir + '/' + tgt_path.split('/')[-1] + '.' + aligner + name + '-align.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class CobaltAligner(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'cobalt_aligner')

    def run(self, config, sample):

        tgt_path = config.get('Data', 'tgt') + '.parse' + '.' + sample
        ref_path = config.get('Data', 'ref') + '.parse' + '.' + sample
        align_dir = config.get('Alignment', 'dir')

        if os.path.exists(align_dir + '/' + tgt_path.split('/')[-1] + '.cobalt-align.out'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        aligner = Aligner('english')
        aligner.align_documents(tgt_path, ref_path)
        aligner.write_alignments(align_dir + '/' + tgt_path.split('/')[-1] + '.cobalt-align.out')

    def get(self, config, sample):

        tgt_path = config.get('Data', 'tgt') + '.parse' + '.' + sample
        align_dir = config.get('Alignment', 'dir')
        reader = CobaltAlignReader()

        result = reader.read(align_dir + '/' + tgt_path.split('/')[-1] + '.' + 'cobalt' + '-align.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class Tokenizer(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'tokenizer')

    def run(self, config, sample):

        method = config.get('Tokenizer', 'method')

        if method == 'aligner':
            aligner = config.get('Alignment', 'aligner')
            if aligner == 'meteor':
                self.tokenize_from_aligner(config, sample)
            elif aligner == 'cobalt':
                self.tokenize_from_parse(config, sample)
            else:
                print "Aligner is not defined"
        elif method == 'quest':
            self.tokenize_quest(config, sample)
        elif method == 'tokenized':
            self.rewrite_tokenized(config, sample)
        elif method == '':
            print "Files are already tokenized!"
        else:
            print "Tokenizer is not defined"


    def tokenize_from_aligner(self, config, sample):

        align_dir = config.get('Alignment', 'dir')
        aligner = 'meteor'
        # src_path = config.get('Data', 'src') + '.' + sample
        # ref_path = config.get('Data', 'ref') + '.' + sample

        tgt_path = config.get('Data', 'tgt') + '.' + sample
        input_file = align_dir + '/' + tgt_path.split('/')[-1] + '.' + aligner + '-align.out'
        output_file_src = config.get('Data', 'src') + '.token.' + sample
        output_file_tgt = config.get('Data', 'tgt') + '.token.' + sample
        output_file_ref = config.get('Data', 'ref') + '.token.' + sample

        # if os.path.exists(output_file_tgt):
        #     print("The file " + output_file_tgt + " is already tokenized.\n Tokenizer will not run.")
        #     return

        o_src = codecs.open(output_file_src, 'w', 'utf-8')
        o_tgt = codecs.open(output_file_tgt, 'w', 'utf-8')
        o_ref = codecs.open(output_file_ref, 'w', 'utf-8')
        lines = codecs.open(input_file, 'r', 'utf-8').readlines()

        for i, line in enumerate(lines):
            if line.startswith('Line2Start:Length'):
                continue

            if line.startswith('Alignment\t'):
                words_test = lines[i + 1].strip().split(' ')
                words_ref = lines[i + 2].strip().split(' ')
                o_src.write(' '.join(words_test) + '\n')
                o_tgt.write(' '.join(words_test) + '\n')
                o_ref.write(' '.join(words_ref) + '\n')

        o_src.close()
        o_tgt.close()
        o_ref.close()

    def tokenize_from_parse(self, config, sample):

        input_file_tgt = config.get('Data', 'tgt') + '.parse' + '.' + sample
        input_file_ref = config.get('Data', 'ref') + '.parse' + '.' + sample
        out_tgt = config.get('Data', 'tgt') + '.' + 'token' + '.' + sample
        out_ref = config.get('Data', 'ref') + '.' + 'token' + '.' + sample
        out_src = config.get('Data', 'src') + '.' + 'token' + '.' + sample

        if os.path.exists(out_tgt):
            print("The file " + out_tgt + " is already tokenized.\n Tokenizer will not run.")
            return

        o_ref = codecs.open(out_ref, 'w', 'utf-8')
        o_tgt = codecs.open(out_tgt, 'w', 'utf-8')
        o_src = codecs.open(out_src, 'w', 'utf-8')
        lines_tgt = codecs.open(input_file_tgt, 'r', 'utf-8')
        lines_ref = codecs.open(input_file_ref, 'r', 'utf-8')
        processed_tgt = read_parsed_sentences(lines_tgt)
        processed_ref = read_parsed_sentences(lines_ref)

        for i, sentence in enumerate(processed_tgt):
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

            o_tgt.write(' '.join(words) + '\n')
            o_src.write(' '.join(words) + '\n')

        for i, sentence in enumerate(processed_ref):
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

            o_ref.write(' '.join(words) + '\n')

        o_ref.close()
        o_tgt.close()
        o_src.close()

    def tokenize_quest(self, config, sample):

        # Using quest tokenizer

        tokenizer = config.get('Tokenizer', 'tool_path')
        input_file = config.get('Data', 'tgt') + '.' + sample
        myinput = open(input_file, 'r')

        myoutput = open(config.get('Data', 'tgt') + '.' + sample + '.token', 'w')
        p = subprocess.Popen(['perl', tokenizer], stdin=myinput, stdout=myoutput)
        p.wait()
        myoutput.flush()

        print "Tokenization finished!"

    def rewrite_tokenized(self, config, sample):

        # When files are already tokenized

        self.rewrite(config.get('Data', 'tgt'), sample)
        self.rewrite(config.get('Data', 'ref'), sample)

        print "Tokenization finished!"

    @staticmethod
    def rewrite(fname, sample):
        myinput = open(fname + '.' + sample, 'r')
        myoutput = open(fname + '.token' + '.' + sample, 'w')
        for line in myinput:
            myoutput.write(line)
        myinput.close()
        myoutput.close()

    def get(self, config, sample):

        sents_tokens_ref = []
        sents_tokens_tgt = []

        lines_ref = codecs.open(config.get('Data', 'ref') + '.' + 'token' + '.' + sample, 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(config.get('Data', 'tgt') + '.' + 'token' + '.' + sample, 'r', 'utf-8').readlines()
        for i, line in enumerate(lines_tgt):
            sents_tokens_tgt.append(line.strip().split(' '))
            sents_tokens_ref.append(lines_ref[i].strip().split(' '))

        AbstractProcessor.set_result_tgt(self, sents_tokens_tgt)
        AbstractProcessor.set_result_ref(self, sents_tokens_ref)

class QuestSentence(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'quest_sentence')

    def run(self, config, sample):

        # Plain files, no tokenization!
        # Check quest configuration file!

        quest_dir = config.get('Quest', 'path')
        src_lang = config.get('Settings', 'src_lang')
        tgt_lang = config.get('Settings', 'tgt_lang')
        src_path = config.get('Data', 'src') + '.' + sample
        tgt_path = config.get('Data', 'tgt') + '.' + sample
        quest_config = config.get('Quest', 'config') + '/' + 'config.' + 'sl.' + tgt_lang + '.properties'
        out_file = config.get('Quest', 'output') + '/' + 'quest.sl.' + sample + '.out'

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

    def get(self, config, sample):

        # Make sure feature file is the same as stated in quest config file
        # Later change to read features from config file

        features_file = config.get('Quest', 'features_sent')
        output_quest = config.get('Quest', 'output') + '/' + 'quest.sl.' + sample + '.out'
        reader = FeaturesReader()
        features = reader.read_features(features_file)

        sentences = open(output_quest, 'r').readlines()
        result = []

        for sent in sentences:
            feats = {}

            for i, val in enumerate(sent.strip().split('\t')):
                feats[features[i]] = float(val)

            result.append(feats)

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class QuestWord(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'quest_word')

    def run(self, config, sample):

        # Tokenized input files !
        # Check quest configuration file!

        quest_dir = config.get('Quest', 'path')
        src_lang = config.get('Settings', 'src_lang')
        tgt_lang = config.get('Settings', 'tgt_lang')
        src_path = config.get('Data', 'src') + '.' + 'token' + '.' + sample
        tgt_path = config.get('Data', 'tgt') + '.' + 'token' + '.' + sample
        quest_config = config.get('Quest', 'config') + '/' + 'config.' + 'wl.' + tgt_lang + '.properties'
        out_file = config.get('Quest', 'output') + '/' + 'quest.wl.' + sample + '.out'

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

    def get(self, config, sample):

        # Make sure feature file is the same as stated in quest config file
        # Later change to read features from config file

        features_file = config.get('Quest', 'features_word')
        output_quest = config.get('Quest', 'output') + '/' + 'quest.wl.' + sample + '.out'
        input_token = config.get('Data', 'tgt') + '.' + 'token' + '.' + sample

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

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)

    @staticmethod
    def sent_length(file_):

        lengths = []
        sents = open(file_, 'r').readlines()
        for sent in sents:
            lengths.append(len(sent.split(' ')))

        return lengths