import codecs
import sys

from utils.parsed_sentences_loader import ParsedSentencesLoader
from utils.stanford_format import StanfordParseLoader
from utils.conll_format import CONNL
from alignment.context_evidence import ContextEvidence

with codecs.open('data_test/test.parse', 'r', 'utf8') as f:
    text = f.read()

loader = ParsedSentencesLoader()
sentences = loader.load(text)
parsed = []

for sentence in sentences['sentences']:
    parsed.append(StanfordParseLoader.process_parse_result(sentence))


sys.exit()
parsed = CONNL.load('/home/marina/workspace/data/TRJuly/txtfile.output.tok.parse')
print(str(len(parsed)))

with codecs.open('data_test/test.parse.out', 'w', 'utf8') as o:
    for i, sentence in enumerate(parsed):
        o.write('Sentence: {}\n'.format(i + 1))
        for word in sentence:
            o.write('{}\t{}\t{}\t{}\n'.format(word.index, word.form, word.dep, -1 if word.head is None else word.head.index))
        o.write('\n')
o.close()
