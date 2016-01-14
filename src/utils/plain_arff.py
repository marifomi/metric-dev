__author__ = 'MarinaFomicheva'

from wmt import read_tsv


def plain_arff(f_names, f_features, dataset, f_out):

    features = read_tsv(f_features)
    names = sorted(open(f_names, 'r').readlines())
    o = open(f_out, 'w')

    print >>o, '@relation ' + dataset

    for name in names:
        print >>o, '@attribute ' + name.strip() + ' real'

    print >>o, '@attribute ' + 'class' + ' {positive, negative}'
    print >>o, '@data'

    for i, instance in enumerate(features):
        if i % 2 == 0:
            print >>o, ','.join([str(x) for x in instance]) + ',' + 'positive'
        else:
            print >>o, ','.join([str(x) for x in instance]) + ',' + 'negative'

    o.close()
