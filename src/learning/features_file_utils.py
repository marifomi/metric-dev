'''
Created on Aug 29, 2012

@author: desouza
'''
import codecs
import numpy as np
import logging as log
import os
import math
from sklearn.cross_validation import train_test_split

def read_labels_file(path, delim, encoding='utf-8'):
    '''Reads the labels of each column in the training and test files (features 
    and reference files).
    
    @param path: the path of the labels file
    @param delim: the character used to separate the label strings.
    @param encoding: the character encoding used to read the file. 
    Default is 'utf-8'.
    
    @return: a list of strings representing each feature column.
    '''
    labels_file = codecs.open(path, 'r', encoding)
    lines = labels_file.readlines()
    
    if len(lines) > 1:
        log.warn("labels file has more than one line, using the first.")
    
    if len(lines) == 0:
        log.error("labels file is empty: %s" % path)
    
    labels = lines[0].strip().split(delim)
    
    return labels
    
    
def read_reference_file(path, delim, encoding='utf-8', tostring=False):
    """Parses the file that contains the references and stores it in a numpy array.
    
       @param path the path of the file.
       @delim char the character used to separate values.
       
       @return: a numpy array representing each instance response value
    """
    
    # reads the references to a vector
    refs_file = codecs.open(path, 'r', encoding)
    refs_lines = []
    for line in refs_file:
        cols = line.strip().split(delim)
        refs_lines.append(cols[0])

    if tostring:
        refs = np.array(refs_lines, dtype='str')
    else:
        refs = np.asfarray(refs_lines)
    
    return refs


def read_features_file(path, delim, encoding='utf-8', tostring=False):
    '''
    Reads the features for each instance and stores it on an numpy array.
    
    @param path: the path to the file containing the feature set.
    @param delim: the character used to separate the values in the file pointed by path.
    @param encoding: the character encoding used to read the file.
    
    @return: an numpy array where the columns are the features and the rows are the instances.
    '''
    # this method is memory unneficient as all the mtc is kept in memory
    feats_file = codecs.open(path, 'r', encoding='utf-8')
    feats_lines = []
    line_num = 0
    for line in feats_file:
        if line == "":
            continue
        toks = tuple(line.strip().split(delim))
        cols = []
        for t in toks:
            if t != '':
                try:
                    if tostring:
                        cols.append(t)
                    else:
                        cols.append(float(t))
                except ValueError as e:
                    log.error("%s line %s: %s" % (e, line_num, t))
        
        line_num += 1
        feats_lines.append(cols)
    
    #    print feats_lines
    feats = np.asarray(feats_lines)
    
    return feats


def split_dataset(input_path_x, input_path_y, output_dir):

    with open(os.path.expanduser(input_path_x), 'r') as f:
        read_data_x = f.readlines()
    f.close()

    with open(os.path.expanduser(input_path_y), 'r') as f:
        read_data_y = f.readlines()
    f.close()

    x_train, x_test, y_train, y_test = train_test_split(read_data_x, read_data_y)

    write_lines_to_file(output_dir + '/' + 'x_train' + '.' + 'tsv', x_train)
    write_lines_to_file(output_dir + '/' + 'y_train' + '.' + 'tsv', y_train)
    write_lines_to_file(output_dir + '/' + 'x_test' + '.' + 'tsv', x_test)
    write_lines_to_file(output_dir + '/' + 'y_test' + '.' + 'tsv', y_test)


def split_dataset_repeated_segments(input_path_x, input_path_y, output_dir, number_of_segments):

    with open(os.path.expanduser(input_path_x), 'r') as f:
        read_data_x = f.readlines()
    f.close()

    with open(os.path.expanduser(input_path_y), 'r') as f:
        read_data_y = f.readlines()
    f.close()

    segment_numbers = range(0, len(read_data_x))
    number_of_batches = int(len(read_data_x)/number_of_segments)
    train_length = int(round(number_of_segments * 80 / 100))
    test_length = int(round(number_of_segments * 20 / 100))

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(number_of_batches):
        print('\n'.join([str(x + 1) for x in segment_numbers[i * number_of_segments + train_length:i * number_of_segments + train_length + test_length]]))
        x_train += read_data_x[i * number_of_segments:i * number_of_segments + train_length]
        y_train += read_data_y[i * number_of_segments:i * number_of_segments + train_length]
        x_test += read_data_x[i * number_of_segments + train_length:i * number_of_segments + train_length + test_length]
        y_test += read_data_y[i * number_of_segments + train_length:i * number_of_segments + train_length + test_length]

    write_lines_to_file(output_dir + '/' + 'x_train' + '.' + 'tsv', x_train)
    write_lines_to_file(output_dir + '/' + 'y_train' + '.' + 'tsv', y_train)
    write_lines_to_file(output_dir + '/' + 'x_test' + '.' + 'tsv', x_test)
    write_lines_to_file(output_dir + '/' + 'y_test' + '.' + 'tsv', y_test)


def concatenate_features_files(file_paths):

    feature_arrays = []
    for fp in file_paths:
        feature_arrays.append(read_features_file(fp, "\t"))

    return np.concatenate(feature_arrays, axis=1)

def write_reference_file(output_path, labels):

    output_file = codecs.open(output_path, 'w', 'utf-8')
    for l in labels:
        output_file.write(str(l) + '\n')
    output_file.close()

def write_feature_file(output_path, feature_matrix):

    output_file = codecs.open(output_path, 'w', 'utf-8')
    for row in feature_matrix:
        output_file.write('\t'.join([str(x) for x in row]) + '\n')
    output_file.close()

def write_lines_to_file(file_path, lines):

    with open(os.path.expanduser(file_path), 'w') as f:
        for line in lines:
            f.write(line)
    f.close()

def combine_alignment_files(language_pairs, directory, file_name):

    # Method to combine alignment files for different languages in a single file

    output_file = codecs.open(directory + "/" + "full_dataset/" + file_name, "w", "utf-8")

    count = 0

    for language_pair in language_pairs:
        lines = codecs.open(directory + "/" + language_pair + "/" + "we" + "/" + file_name, "r", "utf-8")

        for line in lines:
            if "Sentence #" in line:
                count += 1
                output_file.write("Sentence #" + str(count) + "\n")
            else:
                output_file.write(line)

    output_file.close()


def create_single_metrics():

    # Method to combine alignment files for different languages in a single file

    my_dir = os.path.expanduser("~/Dropbox/informative_features_for_evaluation/data")
    metrics_file = my_dir + "/" + "x_newstest2014.metrics.simple.all.tsv"

    bleu_file = open(my_dir + "/" + "x_newstest2014.bleu.all.tsv", "w")
    cobalt_file = open(my_dir + "/" + "x_newstest2014.cobalt.all.tsv", "w")
    meteor_file = open(my_dir + "/" + "x_newstest2014.meteor.all.tsv", "w")

    metrics_feature_values = read_features_file(metrics_file, "\t")

    for i, instance in enumerate(metrics_feature_values):
        bleu_file.write('\t'.join([str(x) for x in instance[:2]]) + "\n")
        cobalt_file.write('\t'.join([str(x) for x in instance[2:4]]) + "\n")
        meteor_file.write('\t'.join([str(x) for x in instance[4:]]) + "\n")

    bleu_file.close()
    cobalt_file.close()
    meteor_file.close()


def write_files_with_selected_features(selected_features_indexes):

    # Inputs the indexes of the selected features 0-indexed

    my_dir = os.path.expanduser("~/Dropbox/informative_features_for_evaluation/data")
    input_file = my_dir + "/" + "x_newstest2015.fluency_features_pos.all.tsv"
    output_file = open(my_dir + "/" + "combination_analysis" + "/" + "x_newstest2015.fluency.word.level.pos.all.tsv", "w")

    feature_values = read_features_file(input_file, "\t")

    for instance in feature_values:

        for i, feature_idx in enumerate(selected_features_indexes):
            resulting = feature_idx * 2

            if i == len(selected_features_indexes) - 1:
                end = "\n"
            else:
                end = "\t"

            output_file.write('\t'.join([str(instance[resulting]), str(instance[resulting + 1])]) + end)

    output_file.close()

def get_number_features():

    my_dir = os.path.expanduser("~/Dropbox/informative_features_for_evaluation/data")
    input_file = my_dir + "/" + "x_newstest2015.metrics_comb_fluency_comb_diff.all.tsv"

    features_values = read_features_file(input_file, "\t")

    print(features_values.shape)

def convert_concatenated_to_difference():

    my_dir = os.path.expanduser("~/Dropbox/informative_features_for_evaluation/data")
    input_file = my_dir + "/" + "x_newstest2015.metrics.comb.fluency.comb.all.tsv"
    output_file = my_dir + "/" + "x_newstest2015.metrics_comb_fluency_comb_diff.all.tsv"

    features_values = read_features_file(input_file, "\t")

    differences_values = []

    for i, sentence in enumerate(features_values):

        sentence_feature_values = []

        for k, feature in enumerate(sentence):

            if is_even(k):
                continue
            else:
                sentence_feature_values.append(feature - sentence[k - 1])

        differences_values.append(sentence_feature_values)

    write_feature_file(output_file, differences_values)

def is_even(number):
    return number % 2 == 0

if __name__ == '__main__':

    # convert_concatenated_to_difference()
    get_number_features()
    # my_dir = os.path.expanduser("~/Dropbox/informative_features_for_evaluation/data/combination_analysis")
    #
    # paths = [my_dir + "/" + "x_newstest2015.surface.features.all.tsv",
    #          my_dir + "/" + "x_newstest2015.linguistic.features.all.tsv",
    #          ]
    # path_to_out = my_dir + "/" + "x_newstest2015.fluency.features.all.all.tsv"
    #
    # feature_arrays = concatenate_features_files(paths)
    #
    # write_feature_file(path_to_out, feature_arrays)


