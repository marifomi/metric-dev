import codecs

class AlignmentsReader(object):

    def read(self, alignment_file):

        alignments = []
        lines = codecs.open(alignment_file, 'r', 'utf-8')

        for line in lines:
            if line.startswith('Sentence #'):
                phrase = int(line.replace('Sentence #', ''))

                if phrase > 1:
                    alignments.append([indexes, words, differences])

                indexes = []
                words = []
                differences = []

            elif line.startswith('['):
                indexes.append(AlignmentsReader.read_alignment_idx(line))
                words.append(AlignmentsReader.read_alignment_words(line))
                differences.append(AlignmentsReader.read_differences(self, line))

        alignments.append([indexes, words, differences])

        lines.close()

        return alignments

    def read_differences(self, line):
        values = line.strip().split(' : ')
        contextInfo = {}

        contextInfo['srcDiff'] = self.my_split(values[2].split(';')[0].split('=')[1])
        contextInfo['srcCon'] = self.my_split(values[2].split(';')[1].split('=')[1])
        contextInfo['tgtDiff'] = self.my_split(values[2].split(';')[2].split('=')[1])
        contextInfo['tgtCon'] = self.my_split(values[2].split(';')[3].split('=')[1])

        return contextInfo

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

