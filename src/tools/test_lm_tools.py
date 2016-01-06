__author__ = 'MarinaFomicheva'

import os.path
from ConfigParser import ConfigParser

from src.utils.core_nlp_utils import *
from src.alignment.aligner import Aligner




# path = '~/workspace/srilm-1.7.1/bin/macosx'
# tokenizer = os.path.expanduser('~/Dropbox/workspace/questplusplus/lang_resources/tokenizer/tokenizer.perl')
# input_file_name = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt13-graham/plain/system-outputs/newstest2013-graham/all-en/newstest2013-graham.all-en.system')
# lm = LangModelGenerator()
# lm.tokenize(tokenizer, input_file_name)

config = ConfigParser()
config.readfp(open(os.path.expanduser('~/workspace/upf-cobalt/config/metric.cfg')))
tst_file_name = os.path.expanduser('~/workspace/upf-cobalt/data/test')
ref_file_name = os.path.expanduser('~/workspace/upf-cobalt/data/reference')
tst_file = open(tst_file_name, 'r')
ref_file = open(ref_file_name, 'r')
tst_phrases = read_sentences(tst_file)
ref_phrases = read_sentences(ref_file)

ppdb_file_name = config.get('Resources', 'ppdb_file_name')
vectors_file_name = config.get('Resources', 'vectors_file_name')

#load_resources.load_ppdb(ppdb_file_name)
#load_resources.load_word_vectors(vectors_file_name)

aligner = Aligner('english')
#aligner.align_documents(tst_phrases, ref_phrases)
#aligner.write_alignments(tst_file_name + '.alignments')
alignment_file = open(os.path.expanduser('~/workspace/upf-cobalt/data/test.alignments'), 'r')
alignments = aligner.read_alignments(alignment_file)
print()








