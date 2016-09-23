import codecs
import os

from os.path import expanduser as usr
from utils import wmt


def write_parsed(input_dir, output_dir, lps):

    print("Copying parsed files to " + output_dir + ' ...')

    path_tgt = usr(output_dir + '/' + 'tgt' + '.parse')
    path_ref = usr(output_dir + '/' + 'ref' + '.parse')

    with codecs.open(path_tgt, 'w', 'utf8') as output_tgt:
        with codecs.open(path_ref, 'w', 'utf8') as output_ref:

            counter_tgt = 0
            counter_ref = 0

            for dataset in sorted(os.listdir(input_dir + '/' + 'references')):
                if dataset.startswith('.'):
                    continue

                for lp in sorted(os.listdir(input_dir + '/' + 'system-outputs' + '/' + dataset)):
                    if lp.startswith('.'):
                        continue
                    if lp not in lps:
                        continue

                    with codecs.open(wmt.reference_path(input_dir, dataset, lp) + '.out', 'r', 'utf8') as input_ref:
                        ref_lines = input_ref.readlines()

                    for sys_file_name in sorted(os.listdir(input_dir + '/' + 'system-outputs' + '/' + dataset + '/' + lp)):
                        if sys_file_name.startswith('.'):
                            continue

                        with codecs.open(input_dir + '/' + 'system-outputs' + '/' + dataset + '/' + lp + '/' + sys_file_name, 'r', 'utf8') as input_sys:

                            for line in input_sys.readlines():
                                if line.startswith('Sentence #'):
                                    counter_tgt += 1
                                    output_tgt.write(wmt.substitute_line_number(line, counter_tgt))
                                else:
                                    output_tgt.write(line)

                            for line in ref_lines:
                                if line.startswith('Sentence #'):
                                    counter_ref += 1
                                    output_ref.write(wmt.substitute_line_number(line, counter_ref))
                                else:
                                    output_ref.write(line)