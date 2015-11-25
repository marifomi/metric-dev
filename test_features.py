__author__ = 'MarinaFomicheva'

import argparse
from training.feature_extractor import *
from scorer import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
            description="Use this script to test an example"
            )

    parser.add_argument("--paths",
            help="File with paths to reference and test files",
            metavar="FILE",
            type=str,
            required=True,
            )

    parser.add_argument("--direction",
            help="Translation direction",
            metavar="DIRECTION",
            type=str,
            required=True
            )

    parser.add_argument("--dataset",
            help="Name of the dataset",
            metavar="DATASET",
            type=str,
            required=True
            )
    parser.add_argument("--phrase_number",
            help="Phrase to process",
            default=0,
            type=int,
            required=True
            )

    parser.add_argument("--system_name",
            help="MT system",
            type=str,
            required=True
            )

    parser.add_argument("--alignments",
            help="""for each aligned word pair, writes the indices of the words,
            word forms, lexical similarity and context penalty""",
            action = 'store_true',
            default=False,
            )

    return parser.parse_args()

args = parse_args()

def main():

    config = ConfigParser()
    config.readfp(open(args.paths))

    reference_dir = config.get('Paths', 'reference_dir')
    test_dir = config.get('Paths', 'test_dir')
    ppdbFileName = config.get('Paths', 'ppdbFileName')
    vectorsFileName = config.get('Paths', 'vectorsFileName')

    loadPPDB(ppdbFileName)
    loadWordVectors(vectorsFileName)
    aligner = Aligner('english')

    ref_file_name = norm_ref_file_name(reference_dir, args.dataset, args.direction)
    ref_data = read_sentences(codecs.open(ref_file_name, encoding='UTF-8'))
    testFiles = [f for f in listdir(test_dir + '/' + args.dataset + '/' + args.direction) if isfile(join(test_dir + '/' + args.dataset + '/' + args.direction, f))]

    for t in testFiles:
        system = get_sys_name(t, args.dataset, args.direction)

        if not system == args.system_name:
            continue

        test_data = read_sentences(codecs.open(test_dir + '/' + args.dataset + '/' + args.direction + '/' + t, encoding='UTF-8'))

        for i, phrase_ref in enumerate(ref_data):
            num_phrase = i + 1

            if args.phrase_number != num_phrase:
                continue

            alignments = aligner.align(test_data[i], phrase_ref)
            candidate_parsed = prepareSentence2(test_data[i])
            reference_parsed = prepareSentence2(phrase_ref)

            features_data = FeatureExtractor()
            sentence_features = features_data.compute_features(test_data[i], phrase_ref, candidate_parsed, reference_parsed, alignments)

            print args.direction + '\t' + system + '\t' + str(num_phrase) + '\n' + '\n'.join(sentence_features)

            scorer = Scorer()
            word_level_scores = scorer.word_scores(candidate_parsed, reference_parsed, alignments)
            sentence_level_score = scorer.sentence_score_cobalt(candidate_parsed, reference_parsed, alignments, word_level_scores)

            print('Sentence #' + str(num_phrase))

            if (args.alignments):

                for index in xrange(len(alignments[0])):
                    print(str(alignments[0][index]) + " : " + str(alignments[1][index]) + " : " + str(word_level_scores[index].similarity) + " : " + str(word_level_scores[index].penalty_mean))

            print('Sentence-level metric score is: ' + str(sentence_level_score))


def norm_ref_file_name(ref_dir, dataset, langPair):

    if dataset == 'newstest2013':
        file_name = ref_dir + '/' + dataset + '/' + dataset + '-ref.' + langPair.split('-')[1] + '.out'
    elif dataset == 'newstest2015' or dataset == 'newsdiscusstest2015':
        file_name = ref_dir + '/' + dataset + '/' + dataset + '-' + langPair.split('-')[0] + langPair.split('-')[1] + '-ref.' + langPair.split('-')[1] + '.out'
    elif '2007' in dataset:
        file_name = ref_dir + '/' + dataset + '/' + dataset + '.' + langPair.split('-')[1] + '.out'
    elif 'ce_eamt' in dataset:
        file_name = ref_dir + '/' + dataset + '/' + langPair + '/target_postedited.out'
    else:
        file_name = ref_dir + '/' + dataset + '-ref.' + langPair + '.out'

    return file_name


def get_sys_name(testFileName, dataset, langPair):

    if dataset == 'newstest2013':
        sys_name = re.sub('^%s\.%s\.(?P<name>.+)\.%s$' % (dataset, langPair, 'out'), '\g<name>', testFileName)
    elif '2007' in dataset:
        sys_name = testFileName.split('.')[0]
    elif 'eamt' in dataset:
        sys_name = 'smt'
    else:
        sys_name = testFileName.split('.')[1] + '.' + testFileName.split('.')[2]

    return sys_name

if __name__ == "__main__":
    main()