__author__ = 'MarinaFomicheva'

from collections import defaultdict
from collections import namedtuple
from csv import DictReader
import re


class HumanRank(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def add_human_data(self, f_judgments, directions, max_segments=None):

        counter = 1

        ranks = open(f_judgments, 'r')

        for line in DictReader(ranks):

            if max_segments and counter > max_segments:
                return

            direction = self.get_direction(line)

            if direction not in directions:
                continue

            dataset = self.get_dataset(line)
            segment = self.get_segment(line)
            systems_ranks = self.get_system_ranks(line, dataset, direction)
            systems_ranks.sort(key = lambda x: x.id.lower())

            # Extract all comparisons (Making sure that two systems are extracted only once)
            # Also the extracted relation '<' means "is better than"
            compare = lambda x, y: '<' if x < y else '>' if x > y else '='
            extracted_comparisons = [
                    #(segment, sys1.id, sys2.id, compare(sys1.rank, sys2.rank))
                    HumanComparison(segment, sys1.id, sys2.id, compare(sys1.rank, sys2.rank))
                    for idx1, sys1 in enumerate(systems_ranks)
                    for idx2, sys2 in enumerate(systems_ranks)
                    if idx1 < idx2
                    and sys1.rank != -1
                    and sys2.rank != -1
                ]

            self[direction] += extracted_comparisons
            counter += 1

    def get_direction(self, line):

        if '2013' in line['system1Id']:
            return line['system1Id'].split('.')[1]
        elif '2015' in line['system1Id']:
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