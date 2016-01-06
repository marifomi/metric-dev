__author__ = 'MarinaFomicheva'

from collections import namedtuple
from collections import defaultdict
from csv import DictReader
import re

import numpy

from src.training.human_comparison import *


class HumanDataScores(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def add_wmt12_qe(self, file_like):

        counter = 0

        for line in file_like:
            counter += 1
            scores = line.strip().split('\t')
            self['en-es'].append(HumanScore(counter, 'system', scores[0]))
            self['en-es'].append(HumanScore(counter, 'system', scores[1]))
            self['en-es'].append(HumanScore(counter, 'system', scores[2]))

    def add_eamt(self, file_like):

        counter = 0

        for line in file_like:
            counter += 1
            score = line.strip()
            self['fr-en'].append(HumanScore(counter, 'smt', score))

    def add_wmt2007(self, file_like):

        for line in DictReader(file_like):

            if line['TYPE'] != 'NIST':
                continue

            direction = HumanDataScores.convert_wmt2007(line['TASK'].split(' ')[1])
            systems = [line['SYSTEM_0'], line['SYSTEM_1'], line['SYSTEM_2'], line['SYSTEM_3'], line['SYSTEM_4']]
            segment = int(line['ITEM_ID']) + 1
            scores = [
                (line['SCORE_0_A'], line['SCORE_0_B']),
                (line['SCORE_1_A'], line['SCORE_1_B']),
                (line['SCORE_2_A'], line['SCORE_2_B']),
                (line['SCORE_3_A'], line['SCORE_3_B']),
                (line['SCORE_4_A'], line['SCORE_4_B']),
                ]

            for i, system in enumerate(systems):
                if not system or not scores[i][0].isdigit() or not scores[i][1].isdigit():
                    continue
                score = scores[i][0]  # Adequacy score
                self[direction].append(HumanScore(segment, system, score))

    def add_wmt13_graham(self, file_like):

        for line in DictReader(file_like, delimiter='\t'):

            segment = int(line['SID']) + 1
            score = line['HUMAN']
            system = line['SYSTEM']
            lang_pair = line['LP']
            self[lang_pair].append(HumanScore(segment, system, score))

    def add_human_data_sample(self, file_like):

        count = 0

        for line in file_like:
            count += 1

            self['all-en'].append(HumanScore(count, 'system', float(line.strip())))

    @staticmethod
    def convert_wmt2007(word):

        if word == 'Commentary':
            return 'nc-test2007'
        elif word == 'Europarl':
            return 'test2007'
        else:
            new = re.sub(r'^(?P<firstLan>..).*-(?P<secondLan>..).*$', '\g<firstLan>-\g<secondLan>', word).lower()
            if 'cz' in new:
                return re.sub('cz', 'cs', new)
            if 'sp' in new:
                return re.sub('sp', 'es', new)
            if 'ge' in new:
                return re.sub('ge', 'de', new)
            if 'fr' in new:
                return new

    def list_to_dict(self):

        human_dict = defaultdict(DictList)

        for lang_pair in self.keys():
            for human_score in self[lang_pair]:
                human_dict[lang_pair][human_score.system, human_score.phrase].append(human_score.score)

        return human_dict

    @staticmethod
    def average_multiple_judgments(human_dict):

        avg_human_dict = defaultdict(dict)

        for lang_pair in human_dict.keys():
            for system, phrase in human_dict[lang_pair]:
                avg_human_dict[lang_pair][system, phrase] = numpy.mean(human_dict[lang_pair][system, phrase])

        return avg_human_dict


class HumanDataRank(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def add_human_data(self, judgments, directions, max_segments):

        counter = 1

        for line in DictReader(judgments):

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


class DictList(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)


class DictDict(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, dict)


class HumanScore(object):

    def __init__(self, phrase, sys, score):
        self.phrase = int(phrase)
        self.system = sys
        self.score = float(score)