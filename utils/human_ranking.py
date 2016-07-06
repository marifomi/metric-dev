import re
import numpy as np
import itertools
import os

from collections import defaultdict
from collections import namedtuple
from collections import Counter
from csv import DictReader
from json import loads


class HumanComparison(object):

    def __init__(self, phrase, sys1, sys2, sign):
        self.phrase = phrase
        self.sys1 = sys1
        self.sys2 = sys2
        self.sign = sign

        self.idx_phrase_sys1 = -1
        self.idx_phrase_sys2 = -1


class HumanRanking(defaultdict):

    def __init__(self):
        defaultdict.__init__(self, list)

    def add_human_data(self, config, max_comparisons=-1):

        counter = 1
        ranks = open(os.path.expanduser(config.get('Paths', 'judgments')), 'r')
        lang_pairs = loads(config.get('Settings', 'lang_pairs'))

        for line in DictReader(ranks):

            if counter > max_comparisons > 0:
                return

            lang_pair = HumanRanking.lang_pair(line)
            if lang_pair not in lang_pairs:
                continue

            dataset = HumanRanking.dataset(line)
            segment = HumanRanking.sent_number(line)
            systems_ranks = HumanRanking.system_ranks(line, dataset, lang_pair)
            systems_ranks.sort(key=lambda x: x.id.lower())

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

            self[dataset, lang_pair] += extracted_comparisons
            counter += len(extracted_comparisons)

    def clean_data(self):
        """ Filters out the judgments provided by different judges for the same pair of MT outputs. Majority voting
        is used in order to choose the a single judgment. Cases where different judgments have been assigned
        by the same number of judges are eliminated """

        unique_comparisons = defaultdict(list)
        listed_comparisons = defaultdict(list)

        for dataset, lang_pair in sorted(self.keys()):

            ties = 0

            for comp in self[dataset, lang_pair]:
                listed_comparisons[comp.phrase, comp.sys1, comp.sys2].append(comp.sign)

            for sign in itertools.chain.from_iterable(listed_comparisons.values()):
                if sign == '=':
                    ties += 1

            comparisons_numbers = [len(x) for x in listed_comparisons.values()]
            max_comparison_number = max(comparisons_numbers)
            avg_comparison_number = np.mean(comparisons_numbers)
            total_comparison_number = np.sum(comparisons_numbers)

            for item in listed_comparisons.keys():

                c = Counter(listed_comparisons[item])
                signs_and_counts = list(c.items())
                counts = [x[1] for x in signs_and_counts]

                if len(c) == 1:
                    unique_comparisons[lang_pair].append(
                        HumanComparison(item[0], item[1], item[2], listed_comparisons[item][0]))
                    continue

                if counts.count(max(counts)) > 1:
                    continue

                idx = np.argmax(counts)
                my_sign = signs_and_counts[idx][0]
                unique_comparisons[lang_pair].append(HumanComparison(item[0], item[1], item[2], my_sign))

            print(lang_pair + ' ' + str(max_comparison_number) + ' ' + str(avg_comparison_number) + ' ' + str(
                total_comparison_number) + ' ' + str(ties / float(total_comparison_number)) + ' ' + str(
                len(unique_comparisons[lang_pair])))

        return unique_comparisons

    @staticmethod
    def lang_pair(line):

        if '2013' in line['system1Id']:
            return line['system1Id'].split('.')[1]
        elif '2015' in line['system1Id']:
            return re.sub(r'^.+\.(?P<l1>..)-(?P<l2>..)\.txt$', '\g<l1>-\g<l2>', line['system1Id'])
        else:
            return line['system1Id'].split('.')[-1]

    @staticmethod
    def dataset(line):
        return line['system1Id'].split('.')[0]

    @staticmethod
    def sent_number(line):
        return int(line['srcIndex'])

    @staticmethod
    def system_ranks(line, dataset, direction):
        systems_ranks = []
        SystemsTuple = namedtuple("SystemTuple", ["id", "rank"])

        if '2013' in line['system1Id']:
            extract_system = lambda x: re.sub('^%s\.%s\.(?P<name>.+)$' % (dataset, direction), '\g<name>', x)
        elif '2015' in line['system1Id']:
            extract_system = lambda x: re.sub('^%s\.(?P<name>.+)\.%s\.txt$' % (dataset, direction), '\g<name>', x)
        else:
            extract_system = lambda x: '.'.join(x.split('.')[1:3])

        for number in range(1, 6):
            if 'system' + str(number) + 'Id' in line.keys():
                systems_ranks.append(SystemsTuple(id=extract_system(line['system' + str(number) + 'Id']),
                                                  rank=int(line['system' + str(number) + 'rank'])))

        return systems_ranks
