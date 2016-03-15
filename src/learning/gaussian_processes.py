"""
Simple Gaussian Processes regression with an RBF kernel
"""
import pylab as pb
import numpy as np
import GPy
import scipy

pb.ion()
pb.close('all')

def regression(xtr, xte, ytr, yte, features):

    X = np.genfromtxt(xtr)
    test_X = np.genfromtxt(xte)
    Y = np.genfromtxt(ytr).reshape(-1, 1)
    test_Y = np.genfromtxt(yte).reshape(-1, 1)

    # rescale X
    mx = np.mean(X,axis=0)
    sx = np.std(X,axis=0)

    ok = (sx > 0)
    X = X[:,ok]
    test_X = test_X[:,ok]
    sx, mx = sx[ok], mx[ok]

    X = (X - mx) / sx
    test_X = (test_X - mx) / sx

    print('Dropped features with constant values:')
    print(np.nonzero(ok == False)[0])

    # we could centre Y too?

    D = X.shape[1]

    # this is as big as I can go on my laptop :)
    if False:
        X = X[:1200,:]
        Y = Y[:1200,:]

    # construct kernel
    rbf = GPy.kern.RBF(D, ARD=True)
    noise = GPy.kern.White(D)
    kernel = rbf + noise

    # create simple GP model
    m = GPy.models.GPRegression(X, Y, kernel = kernel)

    ls = m['.*lengthscale']
    m['.*lengthscale'] = ls

    m.constrain_positive('')
    m.optimize(max_f_eval=50, messages=True)
    print(m)

    mu, s2 = m.predict(test_X)
    mae = np.mean(np.abs(mu - test_Y))
    rmse = np.mean((mu - test_Y) ** 2) ** 0.5

    print('SEiso -- mae', mae, 'rmse', rmse)

    # the iso kernel initialises the ARD one to avoid local minima
    m.optimize(max_f_eval=100, messages=True)
    print(m)

    mu, s2 = m.predict(test_X)
    mae = np.mean(np.abs(mu - test_Y))
    rmse = np.mean((mu - test_Y) ** 2) ** 0.5
    pearson = scipy.stats.pearsonr(mu, test_Y)

    print('SEard -- mae', mae, 'rmse', rmse, 'pearson', pearson)

    sorted_ls = np.argsort(m['.*lengthscale'])

    print('Feature ranking by length scale')
    print(sorted_ls)

    if len(sorted_ls) == len(features):
        for i, feature in enumerate(features):
            print(feature + '\t' + str(sorted_ls[i]))
    else:
        for i, feature in enumerate(features):
            print(feature)

def main():

    prefix = 'align_cobalt_word_level'

    xtr = '/Users/MarinaFomicheva/workspace/upf-cobalt/results/wmt13_graham/' + prefix + '_features' + '.train.tsv'
    xte = '/Users/MarinaFomicheva/workspace/upf-cobalt/results/wmt13_graham/' + prefix + '_features' + '.test.tsv'
    ytr = '/Users/MarinaFomicheva/workspace/upf-cobalt/data/wmt13_graham/human.train'
    yte = '/Users/MarinaFomicheva/workspace/upf-cobalt/data/wmt13_graham/human.test'
    ffeatures = '/Users/MarinaFomicheva/workspace/upf-cobalt/config/feature_groups/align_cobalt_word_level'
    f_out = '/Users/MarinaFomicheva/workspace/upf-cobalt/analysis/gaussian/' + prefix + '_ranked'

    features = get_features_to_test(ffeatures)

    regression(xtr, xte, ytr, yte, features, f_out)


def get_features_to_test(fname):
    features_to_extract = []
    f = open(fname, 'r')
    for line in f:
        features_to_extract.append(line.strip())
    return features_to_extract

if __name__ == '__main__':
    main()

