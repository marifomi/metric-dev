import codecs


def process_semeval(path_input, output_dir):

    tgt = []
    ref = []

    with codecs.open(path_input, 'r', 'utf8') as f:

        lines = f.readlines()
        for line in lines:
            tgt.append(line.strip().split('\t')[0])
            ref.append(line.strip().split('\t')[1])

    with codecs.open(output_dir + '/' + path_input.split('/')[-1].replace('.txt', '') + '.tgt.txt', 'w', 'utf8') as o:
        for s in tgt:
            o.write(s + '\n')

    with codecs.open(output_dir + '/' + path_input.split('/')[-1].replace('.txt', '') + '.ref.txt', 'w', 'utf8') as o2:
        for s in ref:
            o2.write(s + '\n')


