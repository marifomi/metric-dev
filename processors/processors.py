import shutil
import numpy
import codecs
import os
import subprocess
import shutil
import numpy as np
import re

from collections import defaultdict
from configparser import ConfigParser
from processors.abstract_processor import AbstractProcessor
from alignment.aligner import Aligner
from alignment.aligner_stanford import AlignerStanford
from alignment.context_info_compiler import ContextInfoCompiler
from alignment.aligner import punctuations
from utils.cobalt_align_reader_stanford import CobaltAlignReaderStanford
from utils.cobalt_align_reader import CobaltAlignReader
from utils.meteor_align_reader import MeteorAlignReader
from utils.prepare_wmt import PrepareWmt
from utils import wmt
from utils.core_nlp_utils import read_parsed_sentences, prepareSentence2, parse_text, dependencyParseAndPutOffsets
from utils.features_reader import FeaturesReader
from utils import txt_xml as xml
from utils.stanford_format import StanfordParseLoader
from gensim.models.word2vec import Word2Vec
from numpy import array
from numpy import zeros
from processors.language_model import LanguageModel
from utils.load_resources import load_ppdb, load_word_vectors
from alignment.aligner_config import AlignerConfig
from lex_resources.config import *
from json import loads


class PosLangModel(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_lang_model')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        tagger = PosTagger()
        tagger.run(config)

        f_in = os.path.expanduser(config.get('Data', 'tgt')) + '.pos.join'
        f_lm = os.path.expanduser(config.get('LangModels', 'pos'))
        f_out = os.path.expanduser(config.get('Data', 'tgt')) + '.pos.join' + '.ppl'
        lm = LanguageModel()
        lm.set_path_to_tools('/Users/MarinaFomicheva/workspace/srilm-1.7.1/bin/macosx/')
        lm.produce_ppl(f_in, f_out, f_lm, 3)

    def get(self, config, from_file=False):

        f_token = open(os.path.expanduser(config.get('Data', 'tgt')) + '.token', 'r')
        f_probs = open(os.path.expanduser(config.get('Data', 'tgt')) + '.pos.join' + '.ppl', 'r')
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


class POSTaggerParse(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_tagger_parse')
        AbstractProcessor.set_output(self, None)

    def run(self, config, from_file=False):

        result_tgt = read_parsed_sentences(codecs.open(os.path.expanduser(config.get('Data', 'tgt') + '.' + 'parse'), 'r', 'utf-8'))
        result_ref = read_parsed_sentences(codecs.open(os.path.expanduser(config.get('Data', 'ref') + '.' + 'parse'), 'r', 'utf-8'))

        output_tgt = codecs.open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'pos', 'w', 'utf-8')
        output_ref = codecs.open(os.path.expanduser(config.get('Data', 'ref')) + '.' + 'pos', 'w', 'utf-8')

        for sent in result_tgt:
            pos_tags = [x.pos for x in prepareSentence2(sent)]
            output_tgt.write(' '.join(pos_tags) + '\n')

        for sent in result_ref:
            pos_tags = [x.pos for x in prepareSentence2(sent)]
            output_ref.write(' '.join(pos_tags) + '\n')

        output_tgt.close()
        output_ref.close()

    def get(self, config, from_file=False):
        pass

    @staticmethod
    def treetagger_stanford_tags(tag):

        if tag == "NNPS":
            return "NPS"

        if tag == "NNP":
            return "NP"

        if tag == "PRP":
            return "PP"

        if tag == "PRP$":
            return "PP$"

        if tag in [".", "?", "!"]:
            return "SENT"

        new_tag = tag.replace("-LRB-", "(")
        new_tag = new_tag.replace("-RRB-", ")")

        return new_tag


class PosTagger(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_tagger')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        self.split_token(config)
        tagger = os.path.expanduser(config.get('PosTagger', 'path'))

        f_in = os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token' + '.' + 'split'
        f_out = open(os.path.expanduser(config.get('Data', 'tgt')) + '.pos', 'w')

        subprocess.call([tagger, f_in], stdout=f_out)
        f_out.close()
        os.remove(f_in)

        self.join_pos(config)

    @staticmethod
    def tokenize_for_pos(config, sample):

        Tokenizer.tokenize_quest(config)
        f_in = os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token' + sample
        lines = codecs.open(f_in, 'r', 'utf-8').readlines()
        f_out = open(f_in, 'w')

        for line in lines:
            n_line = re.sub(r'(\')([a-zA-Z])', r'\1 \2', line)
            nn_line = re.sub(r'([a-zA-Z])(\')', r'\1 \2', n_line)
            f_out.write(nn_line)

        f_out.close()

    def split_token(self, config):

        Tokenizer.tokenize_quest(config)

        f_in = codecs.open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token', 'r', 'utf-8')
        f_out = codecs.open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token' + '.' + 'split', 'w', 'utf-8')

        for line in f_in:
            f_out.write('\n'.join(line.split(' ')))

        f_in.close()
        f_out.close()

    def join_pos(self, config):

        f_pos = os.path.expanduser(config.get('Data', 'tgt')) + '.pos'
        pos_lines = open(f_pos, 'r').readlines()
        f_token = open(os.path.expanduser(config.get('Data', 'tgt')) + '.token', 'r')
        f_out = open(os.path.expanduser(config.get('Data', 'tgt')) + '.pos' + '.' + 'join', 'w')

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
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        print("Loading word vectors")

    def get(self, config, from_file=False):

        lines_ref = codecs.open(os.path.expanduser(config.get('Data', 'ref')) + '.' + 'token', 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token', 'r', 'utf-8').readlines()

        fvectors = os.path.expanduser(config.get('Vectors', 'path'))

        print("Loading word vectors from " + fvectors)
        wv = Word2Vec.load_word2vec_format(fvectors, binary=False)

        print("Finished loading word vectors from " + fvectors)

        print("Building sentence vectors for target...")
        AbstractProcessor.set_result_tgt(self, self.words2vec(lines_tgt, wv))
        print("Finished building sentence vectors for target")
        print("Building sentence vectors for reference...")
        AbstractProcessor.set_result_ref(self, self.words2vec(lines_ref, wv))
        print("Finished building sentence vectors for reference")

        wv = None
        print("Finished getting word vectors")


    @staticmethod
    def words2vec(sents, model):

        # Here vectors are added for all the words to preserve sentence length
        result = []

        for line in sents:
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
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        print("Loading word vectors")

    def get(self, config, from_file=False):

        print("Getting sentence vectors")

        lines_ref = codecs.open(os.path.expanduser(config.get('Data', 'ref')) + '.' + 'token', 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token', 'r', 'utf-8').readlines()

        fvectors = os.path.expanduser(config.get('Vectors', 'path'))
        wv = Word2Vec.load_word2vec_format(fvectors, binary=False)

        AbstractProcessor.set_result_tgt(self, self.sents2vec(lines_tgt, wv))
        AbstractProcessor.set_result_ref(self, self.sents2vec(lines_ref, wv))

        wv = None
        print("Finished getting sentence vectors")


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
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        print("Parse already exist!")

    def get(self, config, from_file=False):

        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))

        result_tgt = read_parsed_sentences(codecs.open(working_dir + '/' + 'tgt.parse', 'r', 'utf-8'))
        result_ref = read_parsed_sentences(codecs.open(working_dir + '/' + 'ref.parse', 'r', 'utf-8'))

        sents_tgt = []
        sents_ref = []

        for sent in result_tgt:
            sents_tgt.append(prepareSentence2(sent))

        for sent in result_ref:
            sents_ref.append(prepareSentence2(sent))

        AbstractProcessor.set_result_tgt(self, sents_tgt)
        AbstractProcessor.set_result_ref(self, sents_ref)


class ParseStanford(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'parse')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        print("Parse already exist!")

    def get(self, config, from_file=False):

        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))

        result_tgt = StanfordParseLoader.parsed_sentences(working_dir + '/' + 'tgt.parse')
        result_ref = StanfordParseLoader.parsed_sentences(working_dir + '/' + 'ref.parse')

        AbstractProcessor.set_result_tgt(self, result_tgt)
        AbstractProcessor.set_result_ref(self, result_ref)


class Parse2(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'parse2')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        print("Parse already exist!")

    def get(self, config, from_file=False):

        result_tgt = read_parsed_sentences(codecs.open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'parse', 'r', 'utf-8'))
        result_ref = read_parsed_sentences(codecs.open(os.path.expanduser(config.get('Data', 'ref')) + '.' + 'parse', 'r', 'utf-8'))

        sents_tgt = []
        sents_ref = []

        for sent in result_tgt:
            sentence_parse_result = parse_text(sent)
            sents_tgt.append(dependencyParseAndPutOffsets(sentence_parse_result))

        for sent in result_ref:
            sentence_parse_result = parse_text(sent)
            sents_tgt.append(dependencyParseAndPutOffsets(sentence_parse_result))

        AbstractProcessor.set_result_tgt(self, sents_tgt)
        AbstractProcessor.set_result_ref(self, sents_ref)


class Bleu(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'bleu')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        if from_file is True:
            print("Feature values will be read from file")
            return

        src_path = os.path.expanduser(config.get('Data', 'src'))
        tgt_path = os.path.expanduser(config.get('Data', 'tgt'))
        ref_path = os.path.expanduser(config.get('Data', 'ref'))

        if not os.path.exists(src_path):
            shutil.copyfile(tgt_path, src_path)

        if os.path.exists(os.path.expanduser(config.get('Metrics', 'dir')) + '/' + tgt_path.split('/')[-1] + '.bleu.scores'):
            print("Bleu scores already exist!")
            return

        if not os.path.exists(ref_path + '.xml'):
            xml.run(src_path, ref_path, tgt_path)

        bleu_path = os.path.expanduser(config.get('Metrics', 'bleu'))
        my_file = os.path.expanduser(config.get('Metrics', 'dir')) + '/' + tgt_path.split('/')[-1] + '.bleu.scores'
        o = open(my_file, 'w')
        subprocess.call(['perl', bleu_path, '-b', '-d', str(2), '-r', ref_path + '.xml',
                         '-t', tgt_path + '.xml',
                         '-s', src_path + '.xml'], stdout=o)
        o.close()

    def get(self, config, from_file=False):

        result = []

        if from_file is True:
            lang_pairs = loads(config.get('Settings', 'lang_pairs'))
            result = wmt.read_wmt_format(os.path.expanduser(config.get("Metrics", "bleu")), lang_pairs)
        else:
            tgt_path = os.path.expanduser(config.get('Data', 'tgt'))
            scores_file = os.path.expanduser(config.get('Metrics', 'dir')) + '/' + tgt_path.split('/')[-1] + '.bleu.scores'
            for line in open(scores_file).readlines():
                if not line.startswith('  BLEU'):
                    continue
                result.append(float(re.sub(r'^.+= ([01]\.[0-9]+).+$', r'\1', line.strip())))

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class CobaltScorer(AbstractProcessor):
    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'cobalt')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=True):
        print("Feature values will be read from file")
        return

    def get(self, config, from_file=True):

        lang_pairs = loads(config.get('Settings', 'lang_pairs'))
        result = wmt.read_wmt_format(os.path.expanduser(config.get("Metrics", "cobalt")), lang_pairs)

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class MeteorScorer(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'meteor')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        if from_file:
            print("Feature values will be read from file")
            return

        wd = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = wd + '/' + 'tgt.txt'
        ref_path = wd + '/' + 'ref.txt'
        meteor = os.path.expanduser(config.get('Paths', 'meteor'))
        lang = config.get('Settings', 'target_language')
        o = open(wd + '/' + 'meteor.scores', 'w')
        subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang, '-norm'], stdout=o)
        o.close()

    def get(self, config, from_file=False):

        wd = os.path.expanduser(config.get('Data', 'working_dir'))
        result = []

        if from_file:
            lang_pairs = loads(config.get('Settings', 'lang_pairs'))
            result = wmt.read_wmt_format(os.path.expanduser(config.get("Metrics", "meteor")), lang_pairs)
        else:
            with open(wd + '/' + 'meteor.scores') as f:
                for line in f.readlines():
                    if not line.startswith('Segment '):
                        continue
                    result.append(float(line.strip().split('\t')[1]))

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class Paraphrases(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'paraphrases')
        AbstractProcessor.set_output(self, None)

    def run(self, config, from_file=False):
        load_ppdb(os.path.expanduser(config.get("Paraphrases", "path")))

    def get(self, config, from_file=False):
        pass

class MeteorAligner(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'meteor_aligner')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.txt'
        ref_path = working_dir + '/' + 'ref.txt'

        if os.path.exists(working_dir + '/' + 'meteor-align.out'):
            print("Meteor alignments already exist!")
            return

        meteor = os.path.expanduser(config.get('Paths', 'meteor'))
        lang = loads(config.get('Settings', 'language_pairs'))[0].split('-')[1]

        if os.path.exists(tgt_path + '.' + 'token'):
            print("Meteor will not run the tokenizer! Data is already tokenized! ")
            subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path + '.' + 'token', ref_path + '.' + 'token', '-l', lang,
                             '-lower', '-writeAlignments', '-f', working_dir + '/' + 'meteor'])
        else:
            print("Meteor will run the tokenizer! Data is not yet tokenized! ")
            subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang,
                         '-norm', '-writeAlignments', '-f', working_dir + '/' + 'meteor'])


    def get(self, config, from_file=False):

        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        reader = MeteorAlignReader()

        result = reader.read(working_dir + '/' + 'meteor-align.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class CobaltAlignerStanford(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'cobalt_aligner_stanford')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.parse'
        ref_path = working_dir + '/' + 'ref.parse'

        if os.path.exists(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford.out'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        targets = StanfordParseLoader.parsed_sentences(tgt_path)
        references = StanfordParseLoader.parsed_sentences(ref_path)

        aligner = AlignerStanford('english')
        alignments = []

        for i, sentence in enumerate(targets):
            alignments.append(aligner.align(sentence, references[i]))

        output = codecs.open(os.path.expanduser(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford.out'), 'w', 'utf-8')

        for i, alignment in enumerate(alignments):
            print('Sentence #' + str(i + 1), file=output)

            for a in sorted(alignment[0], key=lambda x: x[0]):
                output.write('[' + str(targets[i][a[0] - 1].index) + ', ' + str(references[i][a[1] - 1].index) + ']' + ' : ' +
                             '[' + targets[i][a[0] - 1].form + ', ' + references[i][a[1] - 1].form + ']' + ' : ' +
                             alignment[1][(a[0], a[1])] + '\n')

            output.write('\n')
        output.close()

    def get(self, config, from_file=False):
        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.parse'
        ref_path = working_dir + '/' + 'ref.parse'
        reader = CobaltAlignReaderStanford()
        result = reader.read(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class CobaltAlignerContextInfoCompiler(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'cobalt_aligner_context_info_compiler')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):
        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.parse'
        ref_path = working_dir + '/' + 'ref.parse'

        if os.path.exists(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford-context-diff.out'):
            print("Context difference already compiled.\n Context difference compiler will not run.")
            return

        reader = CobaltAlignReaderStanford()

        alignment_result = reader.read(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford.out')
        targets = StanfordParseLoader.parsed_sentences(tgt_path)
        references = StanfordParseLoader.parsed_sentences(ref_path)

        compiler = ContextInfoCompiler('english')
        info = []

        for i, sentence in enumerate(targets):
            info.append(compiler.compile_context_info(sentence, references[i], alignment_result[i][0]))

        output = codecs.open(os.path.expanduser(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford-context-diff.out'), 'w', 'utf-8')

        for i, context_info in enumerate(info):
            print('Sentence #' + str(i + 1), file=output)

            for j, a in enumerate(alignment_result[i][0]):
                output.write('[' + str(targets[i][a[0] - 1].index) + ', ' + str(references[i][a[1] - 1].index) + ']' + ' : ')
                output.write('[' + targets[i][a[0] - 1].form + ', ' + references[i][a[1] - 1].form + ']' + ' : ')
                output.write(alignment_result[i][2][j] + ' : ')
                output.write('srcDiff=' + ','.join(context_info[j]['srcDiff']) + ';')
                output.write('srcCon=' + ','.join(context_info[j]['srcCon']) + ';')
                output.write('tgtDiff=' + ','.join(context_info[j]['tgtDiff']) + ';')
                output.write('tgtCon=' + ','.join(context_info[j]['tgtCon']) + '\n')

            output.write('\n')
        output.close()

    def get(self, config, from_file=False):
        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.parse'
        ref_path = working_dir + '/' + 'ref.parse'
        reader = CobaltAlignReaderStanford()
        result = reader.read(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align-stanford-context-diff.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class CobaltAligner(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'cobalt_aligner')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.parse'
        ref_path = working_dir + '/' + 'ref.parse'

        align_cfg = AlignerConfig('english')

        if 'paraphrases' in align_cfg.selected_lexical_resources:
            load_ppdb(align_cfg.path_to_ppdb)

        if 'distributional' in align_cfg.selected_lexical_resources:
            load_word_vectors(align_cfg.path_to_vectors)


        if os.path.exists(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align.out'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        aligner = Aligner('english')
        aligner.align_documents(tgt_path, ref_path)
        aligner.write_alignments(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align.out')

    def get(self, config, from_file=False):
        working_dir = os.path.expanduser(config.get('Data', 'working_dir'))
        tgt_path = working_dir + '/' + 'tgt.parse'
        ref_path = working_dir + '/' + 'ref.parse'
        reader = CobaltAlignReader()
        result = reader.read(working_dir + '/' + tgt_path.split('/')[-1] + '.' + ref_path.split('/')[-1] + '.cobalt-align.out')
        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class LowerCaser(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'lower_caser')
        AbstractProcessor.set_output(self, None)

    def run(self, config, from_file=False):

        file_names = [os.path.expanduser(config.get('Data', 'src')) + '.token',
                      os.path.expanduser(config.get('Data', 'tgt')) + '.token',
                      os.path.expanduser(config.get('Data', 'ref')) + '.token'
        ]

        for file_name in file_names:
            file_object = codecs.open(file_name, "r", "utf-8")
            lines = file_object.readlines()
            file_object.close()

            file_output = codecs.open(file_name, 'w', 'utf-8')
            for line in lines:
                file_output.write(line.lower())

            file_output.close()

    def get(self, config, from_file=False):
        pass


class Tokenizer(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'tokenizer')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        method = config.get('Tokenizer', 'method')

        if method == 'aligner':
            aligner = config.get('Alignment', 'aligner')
            if aligner == 'meteor':
                self.tokenize_from_aligner(config)
            elif aligner == 'cobalt':
                self.tokenize_from_parse(config)
            else:
                print("Aligner is not defined")
        elif method == 'quest':
            self.tokenize_quest(config)
        elif method == 'parse':
            self.tokenize_from_parse(config)
        elif method == 'tokenized':
            self.rewrite_tokenized(config)
        elif method == '':
            print("Files are already tokenized!")
        else:
            print("Tokenizer is not defined")


    def tokenize_from_aligner(self, config):

        align_dir = os.path.expanduser(config.get('Alignment', 'dir'))
        aligner = 'meteor'

        tgt_path = os.path.expanduser(config.get('Data', 'tgt'))
        input_file = align_dir + '/' + tgt_path.split('/')[-1] + '.' + aligner + '-align.out'
        output_file_src = os.path.expanduser(config.get('Data', 'src')) + '.token'
        output_file_tgt = os.path.expanduser(config.get('Data', 'tgt')) + '.token'
        output_file_ref = os.path.expanduser(config.get('Data', 'ref')) + '.token'

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

        input_file_tgt = os.path.expanduser(config.get('Data', 'tgt') + '.parse')
        input_file_ref = os.path.expanduser(config.get('Data', 'ref') + '.parse')
        out_tgt = os.path.expanduser(config.get('Data', 'tgt') + '.' + 'token')
        out_ref = os.path.expanduser(config.get('Data', 'ref') + '.' + 'token')
        out_src = os.path.expanduser(config.get('Data', 'src') + '.' + 'token')

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
                word = re.sub(r"^$", r"-", item.form)
                word = re.sub("``|''", "\"", word)
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
                word = re.sub(r"^$", r"-", item.form)
                word = re.sub("``|''", "\"", word)
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

        tokenizer = os.path.expanduser(config.get('Tokenizer', 'path'))
        language = config.get('Settings', 'tgt_lang')

        path_output_tgt = os.path.expanduser(config.get('Data', 'tgt')) + '.token'
        path_output_ref = os.path.expanduser(config.get('Data', 'ref')) + '.token'

        if os.path.exists(path_output_tgt) and os.path.exists(path_output_ref):
            print("The file " + path_output_tgt + "already exists\nTokenizer will not run.")
            return

        f_output_tgt = open(os.path.expanduser(config.get('Data', 'tgt')) + '.token', 'w')
        f_output_ref = open(os.path.expanduser(config.get('Data', 'ref')) + '.token', 'w')

        # Process target

        input_tgt = open(os.path.expanduser(config.get('Data', 'tgt')), 'r')

        p = subprocess.Popen(['perl', tokenizer, '-q', '-l', language], stdin=input_tgt, stdout=f_output_tgt)
        p.wait()
        f_output_tgt.flush()

        # Process reference

        input_ref = open(os.path.expanduser(config.get('Data', 'ref')), 'r')

        p = subprocess.Popen(['perl', tokenizer, '-q', '-l', language], stdin=input_ref, stdout=f_output_ref)
        p.wait()
        f_output_ref.flush()

        # Copy source (for quest)
        shutil.copyfile(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token', os.path.expanduser(config.get('Data', 'src')) + '.' + 'token')


        print("Tokenization finished!")

    def rewrite_tokenized(self, config):

        # When files are already tokenized

        self.rewrite(config.get('Data', 'tgt'))
        self.rewrite(config.get('Data', 'ref'))

        print("Tokenization finished!")

    @staticmethod
    def rewrite(fname):
        myinput = open(fname, 'r')
        myoutput = open(fname + '.token', 'w')
        for line in myinput:
            myoutput.write(line)
        myinput.close()
        myoutput.close()

    def get(self, config, from_file=False):

        sents_tokens_ref = []
        sents_tokens_tgt = []

        lines_ref = codecs.open(os.path.expanduser(config.get('Data', 'ref') + '.' + 'token'), 'r', 'utf-8').readlines()
        lines_tgt = codecs.open(os.path.expanduser(config.get('Data', 'tgt') + '.' + 'token'), 'r', 'utf-8').readlines()
        for i, line in enumerate(lines_tgt):
            sents_tokens_tgt.append(line.strip().split(' '))
            sents_tokens_ref.append(lines_ref[i].strip().split(' '))

        AbstractProcessor.set_result_tgt(self, sents_tokens_tgt)
        AbstractProcessor.set_result_ref(self, sents_tokens_ref)


class QuestWord(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'quest_word')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        # Tokenized input files !
        # Check quest configuration file!

        quest_dir = os.path.expanduser(config.get('Quest', 'path'))
        src_lang = config.get('Settings', 'src_lang')
        tgt_lang = config.get('Settings', 'tgt_lang')
        src_path = os.path.expanduser(config.get('Data', 'src')) + '.' + 'token'
        tgt_path = os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token'
        quest_config = os.path.expanduser(config.get('Quest', 'config')) + '/' + 'config.' + 'wl.' + tgt_lang + '.properties'
        out_file = os.path.expanduser(config.get('Quest', 'output')) + '/' + 'quest.wl' + '.out'

        subprocess.call(['java',
                         '-jar',
                         quest_dir + '/' + 'dist/QuEstWordLevel.jar',
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

    def get(self, config, map_backoff=False, from_file=False):

        # Word-level quest features
        # WCE1015 - Language model backoff behavior value
        # WCE1037 - Longest target n-gram length
        # WCE1041 - Backward language model backoff behavior value

        features_file = os.path.expanduser(config.get('Quest', 'features_word'))
        output_quest = os.path.expanduser(config.get('Quest', 'output')) + '/' + 'quest.wl' + '.out'
        input_token = os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token'

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


class QuestSentence(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'quest_sentence')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        # Plain files, no tokenization!
        # Check quest configuration file!

        quest_dir = os.path.expanduser(config.get('Quest', 'path'))
        src_lang = config.get('Settings', 'src_lang')
        tgt_lang = config.get('Settings', 'tgt_lang')
        src_path = os.path.expanduser(config.get('Data', 'src'))
        tgt_path = os.path.expanduser(config.get('Data', 'tgt'))
        quest_config = os.path.expanduser(config.get('Quest', 'config')) + '/' + 'config.' + 'sl.' + tgt_lang + '.properties'
        out_file = os.path.expanduser(config.get('Quest', 'output')) + '/' + 'quest.sl' + '.out'

        # Copy target to dummy source (for quest)
        if not os.path.exists(src_path):
            shutil.copyfile(tgt_path, src_path)

        if 'LowerCaser' in loads(config.get("Resources", "processors")):
            case = 'lower'
            subprocess.call(['java',
                         '-jar',
                         quest_dir + '/' + 'dist/QuEstSentenceLevel.jar',
                         '-lang',
                         src_lang,
                         tgt_lang,
                         '-input',
                         src_path,
                         tgt_path,
                         '-config',
                         quest_config,
                         '-tok',
                         '-case',
                         case,
                         '-output_file',
                         out_file
            ])

        else:
            subprocess.call(['java',
                         '-jar',
                         quest_dir + '/' + 'dist/QuEstSentenceLevel.jar',
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

    def get(self, config, from_file=False):

        # Make sure feature file is the same as stated in quest config file

        features_file = os.path.expanduser(config.get('Quest', 'features_sent'))
        output_quest = os.path.expanduser(config.get('Quest', 'output')) + '/' + 'quest.sl' + '.out'
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


class LanguageModelWordFeatures(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'language_model_word_features')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        tgt_path = os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token'
        output_path = tgt_path + '.' + 'ppl2'
        lm = os.path.expanduser(config.get('Language Model', 'path'))
        ngram_size = config.get('Language Model', 'ngram_size')
        srilm = os.path.expanduser(config.get('Language Model', 'srilm'))

        if os.path.exists(output_path):
            print('File with lm perplexities already exist')
            return

        my_output = open(output_path, 'w')

        SRILM = [srilm + '/' + 'ngram', '-lm', lm, '-order', ngram_size, '-debug', str(2), '-ppl', tgt_path]
        subprocess.check_call(SRILM, stdout=my_output)

    def get(self, config, from_file=False):

        ppl_file = open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token' + '.' + 'ppl2', 'r')

        result = []
        tmp = []

        lines = ppl_file.readlines()

        for i, line in enumerate(lines):
            if line.startswith('\n'):
                result.append(tmp)
                tmp = []
            elif 'p( ' in line:
                if 'p( </s> |' in line:
                    continue
                if 'OOV' in line:
                    tmp.append([np.nan, np.nan])
                elif '<s>' in line:
                    tmp.append([np.nan, np.nan])
                else:
                    ngram = int(re.sub(r'\[([1-3])gram\]', r'\1', line.strip().split(' ')[6]))
                    prob = float(line.strip().split(' ')[7])
                    tmp.append([prob, ngram])
            else:
                continue
        if len(tmp) > 0:
            result.append(tmp)

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)


class LanguageModelSentenceFeatures(AbstractProcessor):

    # Language model sentence features extracted using SRILM: oov, probability, perplexity

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'language_model_sentence_features')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        tgt_path = os.path.expanduser(config.get('Data', 'tgt') + '.' + 'token')
        output_path = tgt_path + '.' + 'ppl'
        lm = os.path.expanduser(config.get('Language Model', 'path'))
        ngram_size = config.get('Language Model', 'ngram_size')
        srilm = os.path.expanduser(config.get('Language Model', 'srilm'))

        if os.path.exists(output_path):
            print('File with lm perplexities already exist')
            return

        my_output = open(output_path, 'w')

        SRILM = [srilm + '/' + 'ngram', '-lm', lm, '-order', ngram_size, '-debug', str(1), '-ppl', tgt_path]
        subprocess.check_call(SRILM, stdout=my_output)

    def get(self, config, from_file=False):

        ppl_file = open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'token' + '.' + 'ppl', 'r')

        result = []
        tmp = []

        for line in ppl_file:
            if line.startswith('file') and 'OOVs' in line:
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


class POSLanguageModelWordFeatures(AbstractProcessor):

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_language_model_word_features')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        tgt_path = os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'pos'
        output_path = tgt_path + '.' + 'ppl2'
        lm = os.path.expanduser(config.get('Language Model', 'pos_path'))
        ngram_size = config.get('Language Model', 'pos_ngram_size')
        srilm = os.path.expanduser(config.get('Language Model', 'srilm'))

        if os.path.exists(output_path):
            print('File with lm perplexities already exist')
            return

        my_output = open(output_path, 'w')

        SRILM = [srilm + '/' + 'ngram', '-lm', lm, '-order', ngram_size, '-debug', str(2), '-ppl', tgt_path]
        subprocess.check_call(SRILM, stdout=my_output)

    def get(self, config, from_file=False):

        ppl_file = open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'pos' + '.' + 'ppl2', 'r')

        result = []
        tmp = []

        lines = ppl_file.readlines()

        for i, line in enumerate(lines):
            if line.startswith('\n'):
                result.append(tmp)
                tmp = []
            elif 'p( ' in line:
                if 'p( </s> |' in line:
                    continue
                if 'OOV' in line:
                    tmp.append([np.nan, np.nan])
                elif '<s>' in line:
                    tmp.append([np.nan, np.nan])
                else:
                    ngram = int(re.sub(r'\[([1-4])gram\]', r'\1', line.strip().split(' ')[6]))
                    prob = float(line.strip().split(' ')[7])
                    tmp.append([prob, ngram])
            else:
                continue
        if len(tmp) > 0:
            result.append(tmp)

        AbstractProcessor.set_result_tgt(self, result)
        AbstractProcessor.set_result_ref(self, result)

class POSLanguageModelSentenceFeatures(AbstractProcessor):

    # Language model sentence features extracted using SRILM: oov, probability, perplexity

    def __init__(self):
        AbstractProcessor.__init__(self)
        AbstractProcessor.set_name(self, 'pos_language_model_sentence_features')
        AbstractProcessor.set_output(self, True)

    def run(self, config, from_file=False):

        tgt_path = os.path.expanduser(config.get('Data', 'tgt') + '.' + 'pos')
        output_path = tgt_path + '.' + 'ppl'
        lm = os.path.expanduser(config.get('Language Model', 'pos_path'))
        ngram_size = config.get('Language Model', 'pos_ngram_size')
        srilm = os.path.expanduser(config.get('Language Model', 'srilm'))

        if os.path.exists(output_path):
            print('File with lm perplexities already exist')
            return

        my_output = open(output_path, 'w')

        SRILM = [srilm + '/' + 'ngram', '-lm', lm, '-order', ngram_size, '-debug', str(1), '-ppl', tgt_path]
        subprocess.check_call(SRILM, stdout=my_output)

    def get(self, config, from_file=False):

        ppl_file = open(os.path.expanduser(config.get('Data', 'tgt')) + '.' + 'pos' + '.' + 'ppl', 'r')

        result = []
        tmp = []

        for line in ppl_file:
            if line.startswith('file') and 'OOVs' in line:
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
