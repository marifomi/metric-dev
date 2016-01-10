__author__ = 'MarinaFomicheva'

from src.alignment.aligner import Aligner
from src.utils.cobalt_align_reader import CobaltAlignReader
import codecs
import os
import subprocess
import shutil
import numpy as np
from src.utils.core_nlp_utils import read_sentences
from src.utils.core_nlp_utils import prepareSentence2
from src.utils import txt_xml
import re
from src.utils.features_reader import FeaturesReader

class RunTools(object):

    @staticmethod
    def meteor(tgt_path, ref_path, lang, my_file, **kwargs):
        if 'alignments' in kwargs.keys():
            prefix = kwargs['alignments']
        else:
            prefix = 'meteor'
        meteor = os.path.expanduser('~/workspace/meteor-1.5/meteor-1.5.jar')
        o = open(my_file, 'w')
        subprocess.call(['java', '-Xmx2G', '-jar', meteor, tgt_path, ref_path, '-l', lang, '-norm', '-writeAlignments', '-f', prefix], stdout=o)
        o.close()

    @staticmethod
    def bleu(src_path, tgt_path, ref_path, my_file):
        bleu = os.getcwd() + '/' + 'src' + '/' + 'tools' + '/' + 'mteval-v13m.pl'
        o = open(my_file, 'w')
        subprocess.call(['perl', bleu, '-b', '-d', str(2), '-r', ref_path, '-t', tgt_path, '-s', src_path], stdout=o)
        o.close()

    def run_aligner(self, tgt_parse, ref_parse, out_dir):

        if os.path.exists(out_dir + '/' + tgt_parse.split('/')[-1] + '.align'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        aligner = Aligner('english')
        aligner.align_documents(tgt_parse, ref_parse)
        aligner.write_alignments(out_dir + '/' + tgt_parse.split('/')[-1] + '.align')

    def get_alignments(self, tgt_file, align_dir):

        reader = CobaltAlignReader()
        return reader.read(align_dir + '/' + tgt_file.split('/')[-1] + '.align')

    def tokenize_from_parse(self, input_file_name, output_file_name):

        if os.path.exists(output_file_name):
            print("The file " + output_file_name + " is already tokenized.\n Tokenizer will not run.")
            return

        o = codecs.open(output_file_name, 'w', 'utf-8')
        lines = codecs.open(input_file_name, 'r', 'utf-8')
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

    def tokenize_quest(self, tokenizer, input_file_name):

        # Using quest tokenizer

        myinput = open(input_file_name, 'r')
        myoutput = open(input_file_name + '.tok', 'w')
        p = subprocess.Popen(['perl', tokenizer], stdin=myinput, stdout=myoutput)
        p.wait()
        myoutput.flush()

        print "Tokenization finished!"

    def run_quest_sent(self, quest_dir, quest_config, src_lang, tgt_lang, src_path, tgt_path, out_file):

        # Plain files, no tokenization!

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

    def run_quest_word(self, quest_dir, quest_config, src_lang, tgt_lang, src_path, tgt_path, out_file):

        # For combining cobalt and quest, check that the number of words in tokenized files is the same that in the
        # files parsed with stanford corenlp

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

    def get_quest_sent(self, output_quest, features_file):

        reader = FeaturesReader()
        features = reader.read_features(features_file)

        sentences = open(output_quest, 'r').readlines()
        result = []

        for sent in sentences:
            feats = {}

            for i, val in enumerate(sent.strip().split('\t')):
                feats[features[i]] = float(val)

            result.append(feats)

        return result

    def get_quest_word(self, input_tok, output_quest, features_file):

        reader = FeaturesReader()
        features = reader.read_features(features_file)

        lengths = self.sent_length(input_tok)
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

        return result

    @staticmethod
    def sent_length(file_):

        lengths = []
        sents = open(file_, 'r').readlines()
        for sent in sents:
            lengths.append(len(sent.split(' ')))

        return lengths

def main():

    # quest_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus'
    # quest_config = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/config/config.my_word-level.properties'
    # src_lang = 'spanish'
    # tgt_lang = 'english'
    # src_path = '/Users/MarinaFomicheva/Dropbox/workspace/dataSets/wmt13-graham/tokenized/reference.tok'
    # tgt_path = '/Users/MarinaFomicheva/Dropbox/workspace/dataSets/wmt13-graham/tokenized/system.tok'
    # out_file = os.getcwd() + '/' + 'quest' + '/' + 'output.txt'
    # tools = RunTools()
    # tools.run_quest_word(quest_dir, quest_config, src_lang, tgt_lang, src_path, tgt_path, out_file)
    # tools.get_quest_word(tgt_path, os.getcwd() + '/' + 'output' + '/' + 'output.txt')

    # my_file = os.getcwd() + '/' + 'data' + '/' + 'system.parse'
    # out = os.getcwd() + '/' + 'data' + '/' + 'system.stan.tok'
    # tools = RunTools()
    # tools.tokenize_from_parse(my_file, out)

    # dataset = 'newstest2015'
    # sys_dir = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt15-data/parsed/system-outputs/newstest2015')
    # ref_dir = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt15-data/parsed/references/newstest2015')
    # outdir = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt15-data/alignments')
    # for lp in os.listdir(sys_dir):
    #     ref_file = ref_dir + '/' + dataset + '-' + lp.split('-')[0] + lp.split('-')[1] + '-ref.' + lp.split('-')[1] + '.out'
    #     if not lp == 'cs-en':
    #         continue
    #     for sys in os.listdir(sys_dir + '/' + lp):
    #         print sys
    #         sys_file = sys_dir + '/' + lp + '/' + sys
    #         tools = RunTools()
    #         tools.run_aligner(sys_file, ref_file, outdir)

    # src_path = os.getcwd() + '/' + 'data' + '/' + 'source.dummy'
    # ref_path = os.getcwd() + '/' + 'data' + '/' + 'reference'
    # tgt_path = os.getcwd() + '/' + 'data' + '/' + 'system'
    # txt_xml.run(src_path, ref_path, tgt_path)
    # tools = RunTools()
    # bleu_output = os.getcwd() + '/' + 'output' + '/' + 'bleu.scores'
    # meteor_output = os.getcwd() + '/' + 'output' + '/' + 'meteor.scores'
    # tools.bleu(src_path + '.xml', tgt_path + '.xml', ref_path + '.xml', bleu_output)
    # tools.meteor(tgt_path, ref_path, 'en', meteor_output)

    ref_path = os.getcwd() + '/' + 'data' + '/' + 'reference'
    tgt_path = os.getcwd() + '/' + 'data' + '/' + 'system'
    meteor_output = os.getcwd() + '/' + 'test' + '/' + 'meteor.scores'
    meteor_alignments = os.getcwd() + '/' + 'test' + '/' + 'meteor.align'
    tools = RunTools()
    tools.meteor(tgt_path, ref_path, 'en', meteor_output, alignments=meteor_alignments)

if __name__ == '__main__':
    main()