import codecs
import sys
from os import listdir
from os.path import isfile, join
from os.path import expanduser
import subprocess
import csv
import re

from utils.alignments_reader import *


home = expanduser("~")
reference_dir = home + '/Dropbox/dataSets/wmt14-metrics-task/baselines/data/parsed/references'
test_dir = home + '/Dropbox/dataSets/wmt14-metrics-task/baselines/data/parsed/system-outputs'
output_dir = home + '/Dropbox/dataSets/wmt14-metrics-task/submissions/MWA/MWA-WORDNET-MIN'
correlation_script_dir = home + '/Dropbox/dataSets/wmt14-metrics-task'
dataset = 'newstest2014'
metric = 'MWA'
language_pair = ""
best_found = open(home + '/Dropbox/dataSets/wmt14-metrics-task/best.values')

def calculate_correlation():
    command = 'python3 ' + correlation_script_dir + '/compute-segment-correlations --judgments ' + \
              correlation_script_dir + '/judgements-2014-05-14.csv --metrics ' + \
              str(output_dir + '/' + 'mwa-best-wordnet-min.' + language_pair + '.' + 'seg.score')

    output = subprocess.check_output(command, shell=True)
    output = output.replace('*', '')
    output = re.sub(r'[ ]+', '\t', output)

    output_file = correlation_script_dir + '/output.csv'
    output_file_w = open(output_file, 'w')
    output_file_w.write(output)
    output_file_w.close()

    fieldnames = ('Metric', 'de-en', 'fr-en', 'hi-en', 'cs-en', 'ru-en', 'en-de', 'en-fr', 'en-hi', 'en-cs', 'en-ru',
                  'Average', 'wmt12', 'wmt13', 'xties')

    reader = csv.DictReader(open(output_file), delimiter='\t', fieldnames=fieldnames)
    reader.next()
    row = reader.next()
    return float(row[language_pair])


def calculate_scores(scorer):
    sentences_ref = read_sentences(codecs.open(reference_dir + '/' + dataset + '-ref.' + language_pair + '.out', encoding='UTF-8'))

    scoring_output_file = open(output_dir + '/' + 'mwa-best-wordnet-min.' + language_pair + '.' + 'seg.score', 'w')

    test_files = [f for f in listdir(test_dir + '/' + dataset + '/' + language_pair) if isfile(join(test_dir + '/' + dataset + '/' + language_pair, f))]

    reader = Reader()

    for t in test_files:
        system = t.split('.')[1] + '.' + t.split('.')[2]
        sentences_test = read_sentences(codecs.open(test_dir + '/' + dataset + '/' + language_pair + '/' + t, encoding='UTF-8'))
        alignment_file = output_dir + '/' + dataset + '.' + system + '.' + language_pair + '.align.out'
        alignments = reader.read(alignment_file)

        for i, sentence in enumerate(sentences_ref):
            phrase = i + 1
            score = scorer.calculateScore(sentences_test[i], sentence, alignments[phrase])
            scoring_output_file.write(str(metric) + '\t' + str(language_pair) + '\t' + str(dataset) + '\t' + str(system) + '\t' + str(phrase) + '\t' + str(score) + '\n')

    scoring_output_file.close()


def evaluate(values):
    scorer = Scorer()
    scorer.exact = values[0]
    scorer.stem = values[1]
    scorer.synonym = values[2]
    scorer.paraphrase = values[3]
    #scorer.related = values[4]
    #scorer.related_threshold = values[5]
    scorer.context_importance = values[4]
    scorer.minimal_aligned_relatedness = values[5]
    scorer.alpha = values[6]
    scorer.delta = values[7]

    calculate_scores(scorer)
    correlation = calculate_correlation()

    if correlation < 0:
        correlation = 0

    return correlation


def main(args):

    language_pairs = ['de-en', 'fr-en', 'hi-en', 'cs-en', 'ru-en']

    global language_pair

    for line in best_found:
        values = map(float, line.split(':')[0].replace('[', '').replace(']', '').split(','))

        result = 0
        results = []
        for pair in language_pairs:
            language_pair = pair
            result += evaluate(values)

        results.append(result / 5)
        print str(values) + ' : ' + str(result / 5)


    print 'Max : ' + str(max(results))





if __name__ == "__main__":
    main(sys.argv[1:])