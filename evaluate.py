__author__ = 'MarinaFomicheva'

import codecs
import getopt
import sys


def load_ppdb(ppdb_file_name):

    global ppdbDict

    count = 0

    ppdb_file = open(ppdb_file_name, 'r')
    for line in ppdb_file:
        if line == '\n':
            continue
        tokens = line.split()
        tokens[1] = tokens[1].strip()
        ppdbDict[(tokens[0], tokens[1])] = 0.6
        count += 1


def load_word_vectors(vectors_file_name):

    global word_vector
    vector_file = open (vectors_file_name, 'r')

    for line in vector_file:
        if line == '\n':
            continue

        match = re.match(r'^([^ ]+) (.+)',line)
        if type(match) is NoneType:
            continue

        word = match.group(1)
        vector = match.group(2)

        word_vector[word] = vector

def main(args):

    reference_file = ''
    test_file = ''
    output_directory = ''
    write_alignments = False
    vectors_file_name = ''

    opts, args = getopt.getopt(args, 'hr:t:v:a:o:', ['reference=', 'test=', 'vectors_file=', 'writealignments=', 'output_directory='])

    for opt, arg in opts:
        if opt == '-h':
            print '-r <reference_file>'
            print '-t <test_file>'
            sys.exit()
        elif opt in ('-r', '--reference'):
            reference_file = arg
        elif opt in ('-t', '--test'):
            test_file = arg
        elif opt in ('-v', '--vectors_file'):
            vectors_file_name = arg
        elif opt in ('-a', '--writealignments'):
            write_alignments = bool(arg)
        elif opt in ('-o', '--output_directory'):
            output_directory = arg

    if len(opts) == 0:
        reference_file = './data_simple/reference'
        test_file = 'data_simple/test'
        write_alignments = True
        output_directory = './data_simple'

    metric = 'upf-cobalt'

    ppdb_file_name = './lex_resources/ppdb-1.0-xxxl-lexical.extended.synonyms.uniquepairs'
    load_ppdb(ppdb_file_name)
    if vectors_file_name: load_word_vectors(vectors_file_name)

    sentences_ref = read_sentences(codecs.open(reference_file, encoding='UTF-8'))
    sentences_test = read_sentences(codecs.open(test_file, encoding='UTF-8'))

    output_scoring = open(output_directory + '/' + metric + '.seg.score', 'w')
    if write_alignments: output_alignment = open(output_directory + '/' + metric + '.align.out', 'w')

    scorer = Scorer()
    aligner = Aligner('english')

    for i, sentence in enumerate(sentences_ref):
        phrase = i + 1

        sentence1 = prepareSentence2(sentences_test[i])
        sentence2 = prepareSentence2(sentence)

        alignments = aligner.align(sentences_test[i], sentence)
        word_level_scores = scorer.word_scores(sentence1, sentence2, alignments)
        score1 = scorer.sentence_score_cobalt(sentence1, sentence2, alignments, word_level_scores)

        if write_alignments:
            output_alignment.write('Sentence #' + str(phrase) + '\n')

            for index in xrange(len(alignments[0])):
                output_alignment.write("{} \t {} \t {:.2f} \n".format(alignments[0][index], alignments[1][index], word_level_scores[index].penalty_mean))

        output_scoring.write(str(phrase) + '\t' + str(score1) + '\n')

    if write_alignments:
        output_alignment.close()

    output_scoring.close()

if __name__ == "__main__":
    main(sys.argv[1:])
