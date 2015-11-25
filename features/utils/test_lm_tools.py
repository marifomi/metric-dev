__author__ = 'MarinaFomicheva'

import os.path
from lang_model_generator import LangModelGenerator

path = '~/workspace/srilm-1.7.1/bin/macosx'
tokenizer = os.path.expanduser('~/Dropbox/workspace/questplusplus/lang_resources/tokenizer/tokenizer.perl')
input_file_name = os.path.expanduser('~/Dropbox/workspace/questplusplus/lang_resources/english/tmp/sample_corpus.en')
lm = LangModelGenerator()
lm.tokenize(tokenizer, input_file_name)









