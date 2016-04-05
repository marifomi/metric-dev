__author__ = 'MarinaFomicheva'

from collections import defaultdict
from ConfigParser import ConfigParser
from src.processors.abstract_processor import AbstractProcessor
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
from src.processors.language_model import LanguageModel
from src.utils.load_resources import load_ppdb, load_word_vectors
from src.alignment.aligner_config import AlignerConfig
from src.lex_resources.config import *
import shutil


class PosLangModel(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_lang_model')

    def run(self, config, sample):

        tagger = PosTagger()
        tagger.run(config, sample)

        f_in = config.get('Data', 'tgt') + '.pos.join' + sample
        f_lm = config.get('LangModels', 'pos')
        f_out = config.get('Data', 'tgt') + '.pos.join' + '.ppl' + sample
        lm = LanguageModel()
        lm.set_path_to_tools('/Users/MarinaFomicheva/workspace/srilm-1.7.1/bin/macosx/')
        lm.produce_ppl(f_in, f_out, f_lm, 3)

    def get(self, config, sample):

        f_token = open(config.get('Data', 'tgt') + '.token' + sample, 'r')
        f_probs = open(config.get('Data', 'tgt') + '.pos.join' + '.ppl' + sample, 'r')
        lines_probs = []

        for line in f_probs:
            if '</s>' in line:
                continue
            if line.startswith('\t'):
                lines_probs.append(line)

        number_tokens = {}
        sentence_probs = defaultdict(list)

        for i, line in enumerate(f_token):
            number_tokens[i] = len(line.strip().split(' '))

        count_pos = 0
        count_sent = 0
        for pos in lines_probs:
            if count_sent == len(number_tokens):
                break
            sentence_probs[count_sent].append(pos)
            count_pos += 1
            if count_pos == number_tokens[count_sent]:
                count_pos = 0
                count_sent += 1

        result = []
        for sent in sorted(sentence_probs.keys()):
            if len(sentence_probs[sent]) > 0:
                probs = []
                for pos in sentence_probs[sent]:

                    if 'OOV' in pos or '-inf' in pos:
                        probs.append(0.0)
                        continue

                    grs = re.match(r'^.+([0-9])gram] ([0-9]\..+) \[.+$', pos)
                    ngram = int(grs.group(1))
                    prob = float(grs.group(2))
                    probs.append(ngram * prob)

                result.append(probs)

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)

class PosTagger(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_tagger')

    def run(self, config, sample):

        self.split_token(config, sample)
        tagger = config.get('PosTagger', 'path')

        f_in = config.get('Data', 'tgt') + '.' + 'token' + '.' + 'split' + sample
        f_out = open(config.get('Data', 'tgt') + '.pos' + sample, 'w')

        subprocess.call([tagger, f_in], stdout=f_out)
        f_out.close()
        os.remove(f_in)

        self.join_pos(config, sample)

    @staticmethod
    def tokenize_for_pos(config, sample):

        Tokenizer.tokenize_quest(config, sample)
        f_in = config.get('Data', 'tgt') + '.' + 'token' + sample
        lines = codecs.open(f_in, 'r', 'utf-8').readlines()
        f_out = open(f_in, 'w')

        for line in lines:
            n_line = re.sub(r'(\')([a-zA-Z])', r'\1 \2', line)
            nn_line = re.sub(r'([a-zA-Z])(\')', r'\1 \2', n_line)
            f_out.write(nn_line)

        f_out.close()

    def split_token(self, config, sample):

        Tokenizer.tokenize_quest(config, sample)

        f_in = codecs.open(config.get('Data', 'tgt') + '.' + 'token' + sample, 'r', 'utf-8')
        f_out = codecs.open(config.get('Data', 'tgt') + '.' + 'token' + '.' + 'split' + sample, 'w', 'utf-8')

        for line in f_in:
            f_out.write('\n'.join(line.split(' ')))

        f_in.close()
        f_out.close()

    def join_pos(self, config, sample):

        f_pos = config.get('Data', 'tgt') + '.pos' + sample
        pos_lines = open(f_pos, 'r').readlines()
        f_token = open(config.get('Data', 'tgt') + '.token' + sample, 'r')
        f_out = open(config.get('Data', 'tgt') + '.pos' + '.' + 'join' + sample, 'w')

        number_tokens = {}
        sentence_tags = defaultdict(list)

        for i, line in enumerate(f_token):
            number_tokens[i] = len(line.strip().split(' '))

        count_pos = 0
        count_sent = 0
        for pos in pos_lines:
            if count_sent == len(number_tokens):
                break
            sentence_tags[count_sent].append(pos)
            count_pos += 1
            if count_pos == number_tokens[count_sent]:
                count_pos = 0
                count_sent += 1

        for sent in sorted(sentence_tags.keys()):
            if len(sentence_tags[sent]) > 0:
                f_out.write(' '.join([x.split('\t')[1] for x in sentence_tags[sent]]) + '\n')

        os.remove(f_pos)

class WordVectors(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'word_vectors')

    def run(self, config):
        print "Loading word vectors"

    def get(self, config):

        lines_ref = codecs.open(config.get('Data', 'ref') + '.' + 'token', 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(config.get('Data', 'tgt') + '.' + 'token', 'r', 'utf-8').readlines()

        fvectors = config.get('Vectors', 'path')

        print "Loading word vectors from " + fvectors
        wv = Word2Vec.load_word2vec_format(fvectors, binary=False)

        print "Finished loading word vectors from " + fvectors

        print "Building sentence vectors for target..."
        AbstractProcessor.set_result_tgt(self, self.words2vec(lines_tgt, wv))
        print "Finished building sentence vectors for target"
        print "Building sentence vectors for reference..."
        AbstractProcessor.set_result_ref(self, self.words2vec(lines_ref, wv))
        print "Finished building sentence vectors for reference"

        wv = None
        print "Finished getting word vectors"


    @staticmethod
    def words2vec(sents, model):

        # Here vectors are added for all the words to preserve sentence length

        result = []

        cnt = 0
        for line in sents:

            cnt += 1

            print str(cnt)

            tokens = [t.lower() for t in line.strip().split(' ')]
            vecs = []

            for token in tokens:
                if token in model.index2word:
                    vecs.append(model[token])
                else:
                    vecs.append(zeros(model.vector_size))

            result.append(array(vecs))

        return result

    @staticmethod
    def word2vec_format(input_path, vocab_size, vector_size, delimiter):

        lines = open(input_path, 'r').readlines()
        output_path = input_path + '.' + 'word2vec'
        if os.path.exists(output_path):
            print("Already exists!")
            return

        output_f = open(output_path, 'w')
        output_f.write(str(vocab_size) + ' ' + str(vector_size) + '\n')
        for line in lines:
            output_f.write(line.replace('\t', ' '))


class SentVector(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'sent_vector')

    def run(self, config):
        print "Loading word vectors"

    def get(self, config):

        print "Getting sentence vectors"

        lines_ref = codecs.open(config.get('Data', 'ref') + '.' + 'token', 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(config.get('Data', 'tgt') + '.' + 'token', 'r', 'utf-8').readlines()

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

            tokens = [t.lower() for t in line.strip().split(' ')]
            vecs = []

            for token in tokens:
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

        result_tgt = read_parsed_sentences(codecs.open(config.get('Data', 'tgt') + '.' + 'parse' + sample, 'r', 'utf-8'))
        result_ref = read_parsed_sentences(codecs.open(config.get('Data', 'ref') + '.' + 'parse' + sample, 'r', 'utf-8'))

        sents_tgt = []
        sents_ref = []

        for sent in result_tgt:
            sents_tgt.append(prepareSentence2(sent))

        for sent in result_ref:
            sents_ref.append(prepareSentence2(sent))

        AbstractProcessor.set_result_tgt(self, sents_tgt)
        AbstractProcessor.set_result_ref(self, sents_ref)


class Bleu(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'bleu')

    def run(self, config, sample):

        src_path = config.get('Data', 'src') + sample
        tgt_path = config.get('Data', 'tgt') + sample
        ref_path = config.get('Data', 'ref') + sample

        if os.path.exists(config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.bleu.scores'):
            print "Bleu scores already exist!"
            return

        if not os.path.exists(ref_path + '.xml'):
            xml.run(src_path, ref_path, tgt_path)

        bleu_path = config.get('Metrics', 'bleu')
        my_file = config.get('Metrics', 'dir') + '/' + tgt_path.split('/')[-1] + '.bleu.scores'
        o = open(my_file, 'w')
        subprocess.call(['perl', bleu_path, '-b', '-d', str(2), '-r', ref_path + '.xml',
                         '-t', tgt_path + '.xml',
                         '-s', src_path + '.xml'], stdout=o)
        o.close()

    def get(self, config, sample):

        result = []
        tgt_path = config.get('Data', 'tgt') + sample
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

        tgt_path = config.get('Data', 'tgt') + sample
        ref_path = config.get('Data', 'ref') + sample

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
        tgt_path = config.get('Data', 'tgt') + sample
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

    def run(self, config):

        tgt_path = config.get('Data', 'tgt')
        ref_path = config.get('Data', 'ref')
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

    def get(self, config):

        tgt_path = config.get('Data', 'tgt')
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

    def run(self, config):

        tgt_path = config.get('Data', 'tgt') + '.' + 'parse'
        ref_path = config.get('Data', 'ref') + '.' + 'parse'
        align_dir = config.get('Alignment', 'dir')

        align_cfg = AlignerConfig('english')

        if os.path.exists(align_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align.out'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        if 'paraphrases' in align_cfg.selected_lexical_resources:
            load_ppdb(align_cfg.path_to_ppdb)

        if 'distributional' in align_cfg.selected_lexical_resources:
            load_word_vectors(align_cfg.path_to_vectors)

        aligner = Aligner('english')
        aligner.align_documents(tgt_path, ref_path)
        aligner.write_alignments(align_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align.out')

    def get(self, config):

        tgt_path = config.get('Data', 'tgt') + '.' + 'parse'
        ref_path = config.get('Data', 'ref') + '.' + 'parse'
        align_dir = config.get('Alignment', 'dir')
        reader = CobaltAlignReader()

        result = reader.read(align_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


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
        elif method == 'tokenized':
            self.rewrite_tokenized(config)
        elif method == '':
            print "Files are already tokenized!"
        else:
            print "Tokenizer is not defined"


    def tokenize_from_aligner(self, config):

        align_dir = config.get('Alignment', 'dir')
        aligner = 'meteor'

        tgt_path = config.get('Data', 'tgt')
        input_file = align_dir + '/' + tgt_path.split('/')[-1] + '.' + aligner + '-align.out'
        output_file_src = config.get('Data', 'src') + '.token'
        output_file_tgt = config.get('Data', 'tgt') + '.token'
        output_file_ref = config.get('Data', 'ref') + '.token'

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

    def tokenize_from_parse(self, config):

        input_file_tgt = config.get('Data', 'tgt') + '.parse'
        input_file_ref = config.get('Data', 'ref') + '.parse'
        out_tgt = config.get('Data', 'tgt') + '.' + 'token'
        out_ref = config.get('Data', 'ref') + '.' + 'token'
        out_src = config.get('Data', 'src') + '.' + 'token'

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

    @staticmethod
    def tokenize_quest(config):

        # Using quest tokenizer

        tokenizer = config.get('Tokenizer', 'path')
        language = config.get('Settings', 'tgt_lang')

        path_output_tgt = config.get('Data', 'tgt') + '.token'
        path_output_ref = config.get('Data', 'ref') + '.token'

        if os.path.exists(path_output_tgt) and os.path.exists(path_output_ref):
            print "The file " + path_output_tgt + "already exists\nTokenizer will not run."
            return

        f_output_tgt = open(config.get('Data', 'tgt') + '.token', 'w')
        f_output_ref = open(config.get('Data', 'ref') + '.token', 'w')

        # Process target

        input_tgt = open(config.get('Data', 'tgt'), 'r')

        p = subprocess.Popen(['perl', tokenizer, '-q', '-l', language], stdin=input_tgt, stdout=f_output_tgt)
        p.wait()
        f_output_tgt.flush()

        # Process reference

        input_ref = open(config.get('Data', 'ref'), 'r')

        p = subprocess.Popen(['perl', tokenizer, '-q', '-l', language], stdin=input_ref, stdout=f_output_ref)
        p.wait()
        f_output_ref.flush()

        # Copy source (for quest)
        shutil.copyfile(config.get('Data', 'tgt') + '.' + 'token', config.get('Data', 'src') + '.' + 'token')


        print "Tokenization finished!"

    def rewrite_tokenized(self, config):

        # When files are already tokenized

        self.rewrite(config.get('Data', 'tgt'))
        self.rewrite(config.get('Data', 'ref'))

        print "Tokenization finished!"

    @staticmethod
    def rewrite(fname):
        myinput = open(fname, 'r')
        myoutput = open(fname + '.token', 'w')
        for line in myinput:
            myoutput.write(line)
        myinput.close()
        myoutput.close()

    def get(self, config):

        sents_tokens_ref = []
        sents_tokens_tgt = []

        lines_ref = codecs.open(config.get('Data', 'ref') + '.' + 'token', 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(config.get('Data', 'tgt') + '.' + 'token', 'r', 'utf-8').readlines()
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
        src_path = config.get('Data', 'src') + sample
        tgt_path = config.get('Data', 'tgt') + sample
        quest_config = config.get('Quest', 'config') + '/' + 'config.' + 'sl.' + tgt_lang + '.properties'
        out_file = config.get('Quest', 'output') + '/' + 'quest.sl' + sample + '.out'

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

        features_file = config.get('Quest', 'features_sent')
        output_quest = config.get('Quest', 'output') + '/' + 'quest.sl' + sample + '.out'
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


class LangModelWordFeatures(AbstractProcessor):

    # Language model word features extracted using SRILM: oov words

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'lang_model_word_features')

    def run(self, config):

        tgt_path = config.get('Data', 'tgt') + '.' + 'token'
        output_path = tgt_path + '.' + 'ppl2'
        lm = config.get('Language Model', 'path')
        ngram_size = config.get('Language Model', 'ngram_size')
        srilm = config.get('Language Model', 'srilm')

        if os.path.exists(output_path):
            print 'File with lm perplexities already exist'
            return

        my_output = open(output_path, 'w')

        SRILM = [srilm + '/' + 'ngram', '-lm', lm, '-order', ngram_size, '-debug', str(2), '-ppl', tgt_path]
        subprocess.check_call(SRILM, stdout=my_output)

    def get(self, config):

        ppl_file = open(config.get('Data', 'tgt') + '.' + 'token' + '.' + 'ppl2', 'r')

        result = []
        tmp = []
        tokens = []
        counter = 0

        lines = ppl_file.readlines()

        for i, line in enumerate(lines):
            if line.startswith('\n') or i == 0:
                result.append(tmp)
                counter = 0
                tmp = []
                if i == 0:
                    tokens = lines[i].strip().split(' ')
                else:
                    tokens = lines[i + 1].strip().split(' ')
            if 'p( ' in line:
                if 'OOV' in line:
                    tmp.append(tokens[counter])
                counter += 1

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class LanguageModelSentenceFeatures(AbstractProcessor):

    # Language model sentence features extracted using SRILM: oov, probability, perplexity

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'lang_model_sentence_features')

    def run(self, config):

        tgt_path = config.get('Data', 'tgt') + '.' + 'token'
        output_path = tgt_path + '.' + 'ppl'
        lm = config.get('Language Model', 'path')
        ngram_size = config.get('Language Model', 'ngram_size')
        srilm = config.get('Language Model', 'srilm')

        if os.path.exists(output_path):
            print 'File with lm perplexities already exist'
            return

        my_output = open(output_path, 'w')

        SRILM = [srilm + '/' + 'ngram', '-lm', lm, '-order', ngram_size, '-debug', str(1), '-ppl', tgt_path]
        subprocess.check_call(SRILM, stdout=my_output)

    def get(self, config):

        ppl_file = open(config.get('Data', 'tgt') + '.' + 'token' + '.' + 'ppl', 'r')

        result = []
        tmp = []

        for line in ppl_file:
            if config.get('Data', 'tgt') in line:
                break
            if 'OOVs' in line:
                tmp.append(int(line.strip().split(' ')[4]))
            if 'logprob' in line:
                tmp += [float(line.strip().split(' ')[3]), float(line.strip().split(' ')[5])]
                result.append(tmp)
                tmp = []

        ppl_file.close()

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


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
        src_path = config.get('Data', 'src') + '.' + 'token'
        tgt_path = config.get('Data', 'tgt') + '.' + 'token'
        quest_config = config.get('Quest', 'config') + '/' + 'config.' + 'wl.' + tgt_lang + '.properties'
        out_file = config.get('Quest', 'output') + '/' + 'quest.wl' + '.out'

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

    def get(self, config, map_backoff=False):

        # Word-level quest features
        # WCE1015 - Language model backoff behavior value
        # WCE1037 - Longest target n-gram length
        # WCE1041 - Backward language model backoff behavior value

        features_file = config.get('Quest', 'features_word')
        output_quest = config.get('Quest', 'output') + '/' + 'quest.wl' + '.out'
        input_token = config.get('Data', 'tgt') + '.' + 'token'

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

                val = int(val.split('=')[1])
                transform = val

                if map_backoff is True:
                    if features[i] == 'WCE1015' or features[i] == 'WCE1041':
                        transform = self.map_backoff(val)

                word_feats[features[i]] = transform

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
    def map_backoff(val):

        if val == 7:
            return 4
        elif 4 < val < 7:
            return 3
        elif 1 < val <= 4:
            return 2
        else:
            return 1


    @staticmethod
    def sent_length(file_):

        lengths = []
        sents = open(file_, 'r').readlines()
        for sent in sents:
            lengths.append(len(sent.split(' ')))

        return lengths


def main():
    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/absolute.cfg'))

    lm = PosLangModel()
    lm.run(cfg, 'train')
    lm.get(cfg, 'train')

if __name__ == '__main__':
    main()