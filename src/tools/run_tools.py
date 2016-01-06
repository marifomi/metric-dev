__author__ = 'MarinaFomicheva'

from src.alignment.aligner import Aligner
from src.alignment.alignments_reader import AlignmentsReader
import codecs
import os
import subprocess
import shutil
import numpy as np

class RunTools(object):

    def run_aligner(self, tgt_parse, ref_parse, out_dir):

        if os.path.exists(out_dir + '/' + tgt_parse.split('/')[-1] + '.align'):
            print("Alignments already exist.\n Aligner will not run.")
            return

        aligner = Aligner('english')
        aligner.align_documents(tgt_parse, ref_parse)
        aligner.write_alignments(out_dir + '/' + tgt_parse.split('/')[-1] + '.align')

    def get_alignments(self, tgt_file, align_dir):

        reader = AlignmentsReader()
        return reader.read(align_dir + '/' + tgt_file.split('/')[-1] + '.align')

    def tokenize(self, tokenizer, input_file_name):

        myinput = open(input_file_name, 'r')
        myoutput = open(input_file_name + '.tok', 'w')
        p = subprocess.Popen(['perl', tokenizer], stdin=myinput, stdout=myoutput)
        p.wait()
        myoutput.flush()

        print "Tokenization finished!"

    def run_quest_sentence(self, quest_dir, quest_config, src_lang, tgt_lang, src_path, tgt_path, out_dir):
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
                         out_dir
        ])
        shutil.rmtree(os.getcwd() + '/' + 'input')

    def run_quest_word(self, quest_dir, quest_config, src_lang, tgt_lang, src_path, tgt_path, out_file):

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

    def get_quest_word(self, input_tok, output_quest):

        lengths = self.sent_length(input_tok)
        words = open(output_quest, 'r').readlines()
        result = []

        cnt_words = 0
        cnt_sentences = 0
        sent_words = []
        for word in words:
            word_feats = [int(x.split('=')[1]) for x in word.strip().split('\t')]
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

    quest_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus'
    quest_config = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/config/config.my_word-level.properties'
    src_lang = 'spanish'
    tgt_lang = 'english'
    src_path = '/Users/MarinaFomicheva/Dropbox/workspace/dataSets/wmt13-graham/tokenized/reference.tok'
    tgt_path = '/Users/MarinaFomicheva/Dropbox/workspace/dataSets/wmt13-graham/tokenized/system.tok'
    out_file = os.getcwd() + '/' + 'quest' + '/' + 'output.txt'
    tools = RunTools()
    tools.run_quest_word(quest_dir, quest_config, src_lang, tgt_lang, src_path, tgt_path, out_file)
    #tools.get_quest_word(tgt_path, os.getcwd() + '/' + 'output' + '/' + 'output.txt')

if __name__ == '__main__':
    main()