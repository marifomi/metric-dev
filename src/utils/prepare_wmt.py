__author__ = 'MarinaFomicheva'

import os
import codecs
from collections import defaultdict
from src.utils.language_codes import *
from configparser import ConfigParser
import re
from json import loads

""" This class processes and prepares the data
    from WMT datasets.
    Reads the data into a data structure of the form
    [data_dir, data_set, lang_pair, system_path, system_name,
    reference_path, counter_start, counter_end].
    Print all wmt data in a single file for source, reference and translations."""


class PrepareWmt(object):

    def __init__(self, data_type='plain'):
        self.data_type = data_type

    @staticmethod
    def read_wmt_format_into_features_data(config, feature_name):

        feature_data = []

        my_file = open(os.path.expanduser(config.get("Metrics", "dir")) + '/' + config.get("WMT", "dataset") + '.' + feature_name + '.' + 'out', 'r')

        for line in my_file:

            feature, data_set, lang_pair, system_name, seg_id, value = line.strip().split('\t')

            if not config.get('WMT', 'directions') == 'None' and lang_pair not in loads(config.get('WMT', 'directions')):
                continue

            if not '-en' in lang_pair:
                continue

            if len(lang_pair.split('-')[0]) == 3:
                source = LANGUAGE_THREE_TO_TWO[lang_pair.split('-')[0]]
                target = LANGUAGE_THREE_TO_TWO[lang_pair.split('-')[1]]
                lang_pair = source + '-' + target

            feature_data.append([feature, data_set, lang_pair, system_name, int(seg_id), float(value)])

        return [x[-1] for x in sorted(feature_data)]

    @staticmethod
    def order_feature_data(feature_data):

        feature_values = []
        for data_set, lang_pair, system_name, seg_id in sorted(feature_data.keys()):
            sentence_feature = []
            for feature in sorted(feature_data[data_set, lang_pair, system_name, seg_id].keys()):
                sentence_feature.append(feature_data[data_set, lang_pair, system_name, seg_id][feature])
            feature_values.append(sentence_feature)

        return feature_values

    def wmt_format_simple(self, config, feature_name, data_set, lang_pair, system_name, scores):

        f_output = os.path.expanduser(config.get('Data', 'output_dir')) + '/' + data_set + '.' + feature_name + '.' + 'out'

        if os.path.exists(f_output):
            print("Feature file already exist.")
            return

        my_output = open(f_output, 'w')


        for i, score in enumerate(scores):

            my_output.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(feature_name, data_set, lang_pair, system_name,
                                                                str(i + 1), str(score)))
        my_output.close()


    def wmt_format(self, config, feature_name, data_set, scores, data_structure):

        f_output = os.path.expanduser(config.get('WMT', 'output_dir')) + '/' + data_set + '.' + feature_name + '.' + 'out'

        if os.path.exists(f_output):
            print("Feature file already exist.")
            return

        my_output = open(f_output, 'w')

        for data_dir, data_set, lang_pair, system_path, system_name, reference_path, counter_start, counter_end in data_structure:

            if not config.get('WMT', 'directions') == 'None' and lang_pair not in loads(config.get('WMT', 'directions')):
                continue

            if not '-en' in lang_pair:
                    continue

            phrase_number = 0
            for score_index in range(counter_start, counter_end):

                score = str(scores[score_index])

                my_output.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(feature_name, data_set, lang_pair, system_name,
                                                                    str(phrase_number + 1), score))
                phrase_number += 1

        my_output.close()

    def run_processor(self, processor, data_structure, output_dir, data_set):

        """ Receives a processor object, runs the commands on the whole wmt dataset,
        writes file with the output in wmt format """

        result = []

        f_output = output_dir + '/' + data_set + '.' + processor.name + '.' + 'out'

        if os.path.exists(output_dir + '/' + data_set + '.' + processor.name + '.' + 'out'):
            print("Feature file already exist.\n Processor will not run.")
            return


        for data_dir, data_set, lang_pair, system_path, system_name, reference_path in data_structure:

            f_tmp_output = data_dir + '/' + processor.name + '_' + system_name
            processor.run(system_path, reference_path, f_tmp_output)
            processor_output = processor.clean(f_tmp_output)

            for i, line in enumerate(processor_output):
                result.append('{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(processor.name, data_set, lang_pair, system_name,
                                                                    str(i + 1), line.strip()))
            os.remove(f_tmp_output)

        my_output = open(f_output, 'w')
        for line in result:
            my_output.write(line + '\n')

    def print_data_set(self, config, data_structure):

        # Writes wmt data set into a single file, ordered by language pair, system, segment

        path_out_tgt = os.path.expanduser(config.get('WMT', 'output_dir') + '/' + 'tgt')
        path_out_ref = os.path.expanduser(config.get('WMT', 'output_dir') + '/' + 'ref')

        if self.data_type == 'parse':
            path_out_tgt += '.' + 'parse'
            path_out_ref += '.' + 'parse'

        if os.path.exists(path_out_tgt) and os.path.exists(path_out_ref):
            print("Data files already exist.\nWMT printer will not run.")
            return

        print("Printing WMT dataset to " + path_out_tgt + " and " + path_out_ref)

        f_out_tgt = codecs.open(path_out_tgt, 'w', 'utf-8')
        f_out_ref = codecs.open(path_out_ref, 'w', 'utf-8')

        counter_tgt = 0
        counter_ref = 0

        for data_dir, data_set, lang_pair, system_path, system_name, reference_path, counter_start, counter_end in data_structure:

            if not config.get('WMT', 'directions') == 'None' and lang_pair not in loads(config.get('WMT', 'directions')):
                continue

            if not '-en' in lang_pair:
                continue

            f_input_tgt = codecs.open(os.path.expanduser(system_path), 'r', 'utf-8')
            f_input_ref = codecs.open(os.path.expanduser(reference_path), 'r', 'utf-8')

            for line in f_input_tgt:

                if self.data_type == 'parse' and line.startswith('Sentence #'):
                    counter_tgt += 1
                    f_out_tgt.write(self.substitute_line_number(line, counter_tgt))
                else:
                    f_out_tgt.write(line)

            for line in f_input_ref:

                if self.data_type == 'parse' and line.startswith('Sentence #'):
                    counter_ref += 1
                    f_out_ref.write(self.substitute_line_number(line, counter_ref))
                else:
                    f_out_ref.write(line)

            f_input_tgt.close()
            f_input_ref.close()

        f_out_tgt.close()
        f_out_ref.close()

    def get_data_structure(self, config):

        # Extracts data structure with the information on the paths to reference and system files

        data_dir = config.get("WMT", "input_dir")
        data_to_process = []
        data_sets = self.get_data_sets(data_dir)

        for data_set in sorted(data_sets):
            lang_pairs = self.get_lang_pairs(data_dir, data_set)

            counter = 0

            for i, lang_pair in enumerate(sorted(lang_pairs)):

                if not config.get('WMT', 'directions') == 'None' and lang_pair not in loads(config.get('WMT', 'directions')):
                    continue

                if not '-en' in lang_pair:
                    continue

                system_paths, system_names = self.get_mt_systems(data_dir, data_set, lang_pair)
                reference_path = self.get_reference_path(data_dir, data_set, lang_pair)

                length = self.dataset_length(reference_path)

                for k, system_path in enumerate(system_paths):
                    counter_start = counter
                    counter += length
                    counter_end = counter
                    data_to_process.append([data_dir, data_set, lang_pair, system_path, system_names[k], reference_path, counter_start, counter_end])

        return data_to_process

    def get_data_structure2(self, config):

        # Extracts data structure without the information on the paths to reference and system files

        data_dir = config.get("WMT", "input_dir")
        data_to_process = []
        data_sets = self.get_data_sets(data_dir)

        for data_set in sorted(data_sets):
            lang_pairs = self.get_lang_pairs(data_dir, data_set)

            for i, lang_pair in enumerate(sorted(lang_pairs)):

                if not config.get('WMT', 'directions') == 'None' and lang_pair not in loads(config.get('WMT', 'directions')):
                    continue

                if not '-en' in lang_pair:
                    continue

                system_paths, system_names = self.get_mt_systems(data_dir, data_set, lang_pair)
                reference_path = self.get_reference_path(data_dir, data_set, lang_pair)

                length = self.dataset_length(reference_path)

                for k, system_path in enumerate(system_paths):

                    for phrase_number in range(length):
                        data_to_process.append([data_set, lang_pair, system_names[k], phrase_number + 1])

        return data_to_process

    def get_mt_systems(self, data_dir, data_set, lang_pair):

        my_directory = data_dir + '/' + 'system-outputs' + '/' + data_set + '/' + lang_pair
        system_paths = []
        system_names = []

        if self.data_type == 'parse':
            my_directory = my_directory.replace('plain', 'parse')

        for f_system in sorted(os.listdir(os.path.expanduser(my_directory))):
            if f_system.startswith('.'):
                continue

            system_paths.append(my_directory + '/' + f_system)
            system_names.append(self.get_system_name(f_system, data_set))

        return system_paths, system_names

    def get_reference_path(self, data_dir, data_set, lang_pair):

        reference_path = str

        my_directory = data_dir + '/' + 'references'

        if data_set == 'newstest2015' or data_set == 'newsdiscusstest2015':
            reference_path = my_directory + '/' + data_set + '/' + data_set + '-' + lang_pair.split('-')[0] + lang_pair.split('-')[1] + '-' + 'ref' + '.' + 'en'

        if data_set == 'newstest2014':
            reference_path = my_directory + '/' + data_set + '/' + data_set + '-' + 'ref' + '.' + lang_pair

        if data_set == 'newstest2013':
            reference_path = my_directory + '/' + data_set + '-' + 'ref' + '.' + lang_pair.split('-')[1]

        if self.data_type == 'parse':
            reference_path = reference_path.replace('plain', 'parse')
            reference_path += '.' + 'out'

        return reference_path

    def dataset_length(self, input_file):

        if self.data_type == 'plain':
            return sum(1 for line in open(os.path.expanduser(input_file)))

        elif self.data_type == 'parse':
            return self.sentence_number_in_parsed_file(os.path.expanduser(input_file))

        else:
            print("Error! Unknown data type!")

    @staticmethod
    def substitute_line_number(line, counter):
        tokens = re.sub(r'^.+(\(.+\):)$\n', r'\1', line)
        return 'Sentence #' + str(counter) + ' ' + tokens + '\n'

    @staticmethod
    def sentence_number_in_parsed_file(input_file):

        counter = 0
        lines = open(input_file, 'r').readlines()

        for line in lines:
            if line.startswith('Sentence #'):
                counter += 1

        return counter

    @staticmethod
    def get_system_name(system_file, data_set):

        system_file = system_file.replace('.txt', '')
        system_file = system_file.replace(data_set, '')

        if data_set == 'newstest2013':
            return '.'.join(system_file.split('.')[2:])

        if data_set == 'newstest2014':
            return '.'.join(system_file.split('.')[1:-1])

        if data_set == 'newstest2015' or data_set == 'newsdiscusstest2015':
            return '.'.join(system_file.split('.')[1:-1])

    @staticmethod
    def get_data_sets(data_dir):
        result = []
        for data_set in sorted(os.listdir(os.path.expanduser(data_dir + '/' + 'system-outputs'))):
            if data_set.startswith('.'):
                continue
            result.append(data_set)
        return result

    @staticmethod
    def get_lang_pairs(data_dir, data_set):

        result = []
        for lang_pair in sorted(os.listdir(os.path.expanduser(data_dir + '/' + 'system-outputs' + '/' + data_set))):
            if lang_pair.startswith('.'):
                continue
            if '-en' not in lang_pair:
                continue
            result.append(lang_pair)
        return result

    @staticmethod
    def get_lang_pair_complete(src, tgt):

        if tgt in LANGUAGE_CODE_TO_NAME.keys():
            tgt = LANGUAGE_CODE_TO_NAME[tgt]

        if not tgt == 'English':
            return None

        if src in LANGUAGE_CODE_TO_NAME.keys():
            src = LANGUAGE_CODE_TO_NAME[src]

        return '{0}-{1}'.format(src, tgt)


def main():

    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/wmt.fluency_features_alignment_quest.cfg'))

    data_dir = os.path.expanduser('~/Dropbox/workspace/dataSets/wmt14-data/parse')
    prepare_wmt = PrepareWmt(data_type='parse')
    data_structure = prepare_wmt.get_data_structure2(data_dir)
    prepare_wmt.print_data_set(cfg, data_structure)

if __name__ == '__main__':
    main()

