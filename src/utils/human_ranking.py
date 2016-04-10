__author__ = 'MarinaFomicheva'

from collections import defaultdict
from collections import namedtuple
from collections import Counter
from csv import DictReader
from src.utils.language_codes import *
import re
import numpy as np
import itertools
import os
from json import loads


class HumanRanking(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def add_human_data(self, f_judgments, config, max_segments=None):

        counter = 1

        ranks = open(os.path.expanduser(f_judgments), 'r')

        directions = loads(config.get('WMT', 'directions')) if config.get('WMT', 'directions') != 'None' else 'None'

        for line in DictReader(ranks):

            if max_segments and counter > max_segments:
                return

            direction = self.get_direction(line)

            if not directions == 'None' and direction not in directions:
                continue

            dataset = self.get_dataset(line)
            segment = self.get_segment(line)
            systems_ranks = self.get_system_ranks(line, dataset, direction)
            systems_ranks.sort(key = lambda x: x.id.lower())

            # Extract all comparisons (Making sure that two systems are extracted only once)
            # Also the extracted relation '<' means "is better than"
            compare = lambda x, y: '<' if x < y else '>' if x > y else '='
            extracted_comparisons = [
                    HumanComparison(segment, sys1.id, sys2.id, compare(sys1.rank, sys2.rank))
                    for idx1, sys1 in enumerate(systems_ranks)
                    for idx2, sys2 in enumerate(systems_ranks)
                    if idx1 < idx2
                    and sys1.rank != -1
                    and sys2.rank != -1
                ]

            self[dataset, direction] += extracted_comparisons
            counter += 1

    def clean_data(self, directions):

        comparisons = defaultdict(list)

        systems_signs = defaultdict(list)

        for direction in directions:

            for comp in self[direction]:
                systems_signs[comp.phrase, comp.sys1, comp.sys2].append(comp.sign)

            ties = 0

            for sign in itertools.chain.from_iterable(systems_signs.values()):
                if sign == '=':
                    ties += 1

            counts = [len(x) for x in systems_signs.values()]
            max_count = max(counts)
            avg_count = np.mean(counts)
            all_counts = np.sum(counts)

            for dpoint in systems_signs.keys():

                c = Counter(systems_signs[dpoint])

                if len(c) == 1:
                    comparisons[direction].append(HumanComparison(dpoint[0], dpoint[1], dpoint[2], systems_signs[dpoint][0]))
                else:
                    competing = False
                    for i, (sign, count) in enumerate(sorted(c.items(), key=lambda x: x[1], reverse=True)):
                        if count == c.items()[i + 1][1]:
                            competing = True
                        break
                    if competing is True:
                        continue

                    counts = c.items()
                    idx = np.argmax([x[1] for x in counts])
                    my_sign = counts[idx][0]
                    comparisons[direction].append(HumanComparison(dpoint[0], dpoint[1], dpoint[2], my_sign))

            print(direction + ' ' + str(max_count) + ' ' + str(avg_count) + ' ' + str(all_counts) + ' ' + str(ties/float(all_counts)) + ' ' + str(len(comparisons[direction])))

        return comparisons

    def get_direction(self, line):

        if '2013' in line['system1Id']:
            return line['system1Id'].split('.')[1]
        elif '2015' in line['system1Id']:
            # src, tgt = re.sub(r'^.+\.(?P<l1>..)-(?P<l2>..)\.txt$', '\g<l1>-\g<l2>', line['system1Id']).split('-')
            # language_pair = LANGUAGE_THREE_TO_TWO[src] + '-' + LANGUAGE_THREE_TO_TWO[tgt]
            return re.sub(r'^.+\.(?P<l1>..)-(?P<l2>..)\.txt$', '\g<l1>-\g<l2>', line['system1Id'])
        else:
            return line['system1Id'].split('.')[-1]

    def get_dataset(self, line):

        if '2013' in line['system1Id']:
            return line['system1Id'].split('.')[0]
        elif '2015' in line['system1Id']:
            return line['system1Id'].split('.')[0]
        else:
            return line['system1Id'].split('.')[0]

    def get_segment(self, line):
        return int(line['srcIndex'])

    def get_system_ranks(self, line, dataset, direction):
        systems_ranks = []
        SystemsTuple = namedtuple("SystemTuple", ["id","rank"])

        if '2013' in line['system1Id']:
            extract_system = lambda x: re.sub('^%s\.%s\.(?P<name>.+)$' % (dataset, direction), '\g<name>', x)
        elif '2015' in line['system1Id']:
            extract_system = lambda x: re.sub('^%s\.(?P<name>.+)\.%s\.txt$' % (dataset, direction), '\g<name>', x)
        else:
            extract_system = lambda x: '.'.join(x.split('.')[1:3])

        for number in range(1, 6):
            if 'system' + str(number) + 'Id' in line.keys():
                systems_ranks.append(SystemsTuple(id = extract_system(line['system' + str(number) + 'Id']), rank = int(line['system' + str(number) + 'rank'])))

        return systems_ranks

    def print_out(self, file_like):
        for direction in self.human_comparisons.keys():
            for test_case in self.human_comparisons[direction]:
                print >>file_like, direction + ',' + ','.join(test_case)


class HumanComparison(object):

    def __init__(self, phrase, sys1, sys2, sign):
        self.phrase = phrase
        self.sys1 = sys1
        self.sys2 = sys2
        self.sign = sign


def main():
    import os
    from configparser import ConfigParser
    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/system.cfg'))
    lps = ['cs-en', 'es-en', 'de-en', 'fr-en', 'ru-en']

    fjudge = cfg.get('Data', 'human')
    human_ranks = HumanRanking()
    human_ranks.add_human_data(fjudge, lps)
    clean_data = human_ranks.clean_data(['cs-en', 'es-en', 'de-en', 'fr-en', 'ru-en'])

if __name__ == '__main__':
    main()
