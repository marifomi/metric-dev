import codecs

from utils.meteor_align_reader import MeteorAlignReader
from utils.stanford_format import StanfordParseLoader
from alignment.context_evidence import ContextEvidence


parsed_target = StanfordParseLoader.parsed_sentences('data_test/tgt.parse')
parsed_ref = StanfordParseLoader.parsed_sentences('data_test/ref.parse')

meteor_alignments = MeteorAlignReader.read('data_test/tgt.meteor-align.out')
alignments = MeteorAlignReader.alignments(meteor_alignments)

context = ContextEvidence()

for i, alignment in enumerate(alignments):
    for word_pair in alignment:
        word_pair.context_difference = context.context_differences(word_pair.left_word,
                                                                   word_pair.right_word,
                                                                   parsed_target[i],
                                                                   parsed_ref[i],
                                                                   meteor_alignments[i][0])