class Reader(object):

    @staticmethod
    def read_alignment_line(line):
        values = line.split(' : ')
        return [
            int(values[0].split(',')[0].replace('[', '')),
            int(values[0].split(',')[1].replace(']', ''))
        ]

    @staticmethod
    def read_difference_line(line):
        values = line.strip().split(': ')
        if values[1] == 'None':
            return values[0], []
        else:
            return values[0], values[1].split(', ')

    def read(self, alignment_file):
        phrase = 0

        alignments = {}

        for line in open(alignment_file):
            if line.startswith('Sentence #'):
                phrase = int(line.replace('Sentence #', ''))
                if phrase > 1:
                    alignments[phrase - 1] = [alignment, [], differences]
                alignment = []
                differences = []
            elif line.startswith('['):
                v = Reader.read_alignment_line(line)
                alignment.append([v[0], v[1]])
                difference = {}
                differences.append(difference)
            else:
                v = Reader.read_difference_line(line)
                difference[v[0]] = v[1]

        alignments[phrase] = [alignment, [], differences]

        return alignments
