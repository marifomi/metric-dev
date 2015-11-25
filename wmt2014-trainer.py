import codecs
import sys
from os import listdir
from os.path import isfile, join
from os.path import expanduser
import subprocess
import csv
import re

from utils.alignments_reader import *
from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators


home = expanduser("~")
reference_dir = home + '/Dropbox/workspace/dataSets/wmt14-metrics-task/baselines/data/parsed/references'
test_dir = home + '/Dropbox/workspace/dataSets/wmt14-metrics-task/baselines/data/parsed/system-outputs'
output_dir = home + '/Dropbox/workspace/dataSets/wmt14-metrics-task/submissions/MWA/training'
correlation_script_dir = home + '/Dropbox/workspace/dataSets/wmt14-metrics-task'
dataset = 'newstest2014'
metric = 'combo4'
language_pairs = ['de-en', 'fr-en', 'hi-en', 'cs-en', 'ru-en']
training_values = open(home + '/Dropbox/workspace/dataSets/wmt14-metrics-task/submissions/MWA/training/training.values', 'w')


def calculate_correlation():
    value = 0.0
    for pair in language_pairs:
        command = 'python3 ' + correlation_script_dir + '/compute-segment-correlations --judgments ' + \
                  correlation_script_dir + '/judgements-2014-05-14.csv --metrics ' + \
                  str(output_dir + '/' + 'training.' + pair + '.' + 'seg.score')

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
        value += float(row[pair])

    return max(value / len(language_pairs), 0)


def calculate_scores(scorer):
    for pair in language_pairs:
        sentences_ref = read_sentences(codecs.open(reference_dir + '/' + dataset + '-ref.' + pair + '.out', encoding='UTF-8'))

        scoring_output_file = open(output_dir + '/' + 'training.' + pair + '.' + 'seg.score', 'w')

        test_files = [f for f in listdir(test_dir + '/' + dataset + '/' + pair) if isfile(join(test_dir + '/' + dataset + '/' + pair, f))]

        reader = Reader()

        for t in test_files:
            system = t.split('.')[1] + '.' + t.split('.')[2]
            sentences_test = read_sentences(codecs.open(test_dir + '/' + dataset + '/' + pair + '/' + t, encoding='UTF-8'))
            alignment_file = output_dir + '/' + metric + '.' + dataset + '.' + system + '.' + pair + '.align.out'
            alignments = reader.read(alignment_file)

            for i, sentence in enumerate(sentences_ref):
                phrase = i + 1
                score = scorer.calculate_score(sentences_test[i], sentence, alignments[phrase])
                scoring_output_file.write(str(metric) + '\t' + str(pair) + '\t' + str(dataset) + '\t' + str(system) + '\t' + str(phrase) + '\t' + str(score) + '\n')

        scoring_output_file.close()


def evaluate(values):
    scorer = Scorer()
    scorer.exact = values[0]
    scorer.stem = values[1]
    scorer.synonym = values[2]
    scorer.paraphrase = values[3]
    scorer.related = values[4]
    scorer.context_importance = values[5]
    scorer.minimal_aligned_relatedness = values[6]
    scorer.alpha = values[7]
    scorer.beta = 4
    scorer.delta = values[8]
    scorer.arguments = values[9]
    scorer.modifiers = values[10]
    scorer.function = values[11]

    calculate_scores(scorer)
    correlation = calculate_correlation()

    training_values.write('[')
    for i, value in enumerate(values):
        training_values.write(str(value))
        if i < len(values):
            training_values.write(', ')
    training_values.write('] : ' + str(correlation) + '\n')

    training_values.flush()

    if correlation < 0:
        correlation = 0

    return correlation


def main(args):

    #initial_values = [1.0, 0.7, 0.8, 0.6, 0.5, 1.0, 0.01, 0.75, 0.7, 1.0, 0.8, 0.3]

    #Genome instance
    genome = G1DList.G1DList(12)
    genome.setParams(rangemin=0.0, rangemax=1.0)
    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    genome.evaluator.set(evaluate)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.nGenerations = 1000

    #Do the evolution
    ga.evolve(10)

    # Best individual
    print ga.bestIndividual()

    training_values.close()

if __name__ == "__main__":
    main(sys.argv[1:])