__author__ = 'MarinaFomicheva'

import codecs
import re
import os


class MeteorAlignReader(object):

    def read(self, alignment_file):

        alignments = []
        lines = codecs.open(alignment_file, 'r', 'utf-8').readlines()

        for i, line in enumerate(lines):
            if line.startswith('Line2Start:Length'):
                continue

            if line.startswith('Alignment\t'):
                words_test = lines[i + 1].strip().split(' ')
                words_ref = lines[i + 2].strip().split(' ')
                phrase = int(re.sub(r'^Alignment\t([0-9]+).+$', r'\1', line.strip()))

                if phrase > 1:
                    alignments.append([self.add_one(indexes), words, matchers])

                indexes = []
                words = []
                matchers = []

            elif re.match('^[0-9]+:', line):
                aligned_inds, modules = MeteorAlignReader.read_alignment_idx(line)
                indexes += aligned_inds
                matchers += modules
                for pair in aligned_inds:
                    wpair = [words_test[pair[0]], words_ref[pair[1]]]
                    words.append(wpair)

        alignments.append([self.add_one(indexes), words, matchers])

        return alignments

    @staticmethod
    def add_one(indexes):

        result = []
        for pair in indexes:
            new_pair = [pair[0] + 1, pair[1] + 1]
            result.append(new_pair)

        return result

    @staticmethod
    def read_alignment_idx(line):
        elem = re.split('\t+', line)

        inds_test = []
        inds_ref = []

        ind_test = int(elem[1].split(':')[0])
        ind_ref = int(elem[0].split(':')[0])

        len_test = int(elem[1].split(':')[1])
        len_ref = int(elem[0].split(':')[1])

        module = int(elem[2])

        for num in range(len_test):
            inds_test.append(ind_test + num)

        for num in range(len_ref):
            inds_ref.append(ind_ref + num)

        result = []
        modules = []
        if len_ref > len_test:
            for i, ind in enumerate(inds_ref):
                if i >= len_test:
                    pair = [inds_test[-1], ind]
                else:
                    pair = [inds_test[i], ind]
                result.append(pair)
        elif len_test > len_ref:
            for i, ind in enumerate(inds_test):
                if i >= len_ref:
                    pair = [ind, inds_ref[-1]]
                else:
                    pair = [ind, inds_ref[i]]
                result.append(pair)
        else:
            for i, ind in enumerate(inds_test):
                pair = [ind, inds_ref[i]]
                result.append(pair)

        for val in result:
            modules.append(module)


        return [result, modules]

def main():

    my_file = os.getcwd() + '/' + 'test' + '/' + 'meteor.align-align.out'
    reader = MeteorAlignReader()
    alignments = reader.read(my_file)
    print

if __name__ == '__main__':
    main()