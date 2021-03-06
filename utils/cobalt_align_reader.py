import codecs
import os


class CobaltAlignReader(object):

    def read(self, alignment_file):

        alignments = []
        lines = codecs.open(os.path.expanduser(alignment_file), 'r', 'utf-8')

        for line in lines:
            if line.startswith('Sentence #'):
                phrase = int(line.strip().replace('Sentence #', ''))

                if phrase > 1:
                    alignments.append([indexes, words, differences])

                indexes = []
                words = []
                differences = []

            elif line.startswith('['):
                indexes.append(CobaltAlignReader.read_alignment_idx(line))
                words.append(CobaltAlignReader.read_alignment_words(line))
                differences.append(CobaltAlignReader.read_differences(self, line))

        alignments.append([indexes, words, differences])

        lines.close()

        return alignments

    def read_differences(self, line):
        values = line.strip().split(' : ')
        context_info = {}

        if len(values) < 3:
            return context_info

        context_info['srcDiff'] = self.my_split(values[2].split(';')[0].split('=')[1])
        context_info['srcCon'] = self.my_split(values[2].split(';')[1].split('=')[1])
        context_info['tgtDiff'] = self.my_split(values[2].split(';')[2].split('=')[1])
        context_info['tgtCon'] = self.my_split(values[2].split(';')[3].split('=')[1])

        return context_info

    @staticmethod
    def read_alignment_idx(line):
        values = line.split(' : ')
        return [
            int(values[0].split(',')[0].replace('[', '')),
            int(values[0].split(',')[1].replace(']', ''))
        ]

    @staticmethod
    def read_alignment_words(line):
        values = line.split(' : ')
        return [
            values[1].split(', ')[0].replace('[', ''),
            values[1].split(', ')[1].replace(']', '')
        ]

    @staticmethod
    def my_split(list):
        return [x for x in list.split(',') if len(x) > 0]
