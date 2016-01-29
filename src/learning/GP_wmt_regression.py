"""
Simple Gaussian Processes regression with an RBF kernel
"""
import pylab as pb
import numpy as np
import GPy
pb.ion()
pb.close('all')

X = np.genfromtxt('/Users/MarinaFomicheva/workspace/upf-cobalt/results/wmt13_graham/cobalt_word_features.train.tsv')
test_X = np.genfromtxt('/Users/MarinaFomicheva/workspace/upf-cobalt/results/wmt13_graham/cobalt_word_features.test.tsv')
Y = np.genfromtxt('/Users/MarinaFomicheva/workspace/upf-cobalt/data/wmt13_graham/human.train').reshape(-1, 1)
test_Y = np.genfromtxt('/Users/MarinaFomicheva/workspace/upf-cobalt/data/wmt13_graham/human.test').reshape(-1, 1)



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
m = GPy.models.GPRegression(X,Y, kernel = kernel)

ls = m['.*lengthscale']
m['.*lengthscale'] = ls

m.constrain_positive('')
m.optimize(max_f_eval = 50, messages = True)
print(m)

mu, s2 = m.predict(test_X)
mae = np.mean(np.abs(mu - test_Y))
rmse = np.mean((mu - test_Y) ** 2) ** 0.5

print('SEiso -- mae', mae, 'rmse', rmse)

# the iso kernel initialises the ARD one to avoid local minima
m.optimize(max_f_eval = 100, messages = True)
print(m)

mu, s2 = m.predict(test_X)
mae = np.mean(np.abs(mu - test_Y))
rmse = np.mean((mu - test_Y) ** 2) ** 0.5

print('SEard -- mae', mae, 'rmse', rmse)

sorted_ls = np.argsort(m['.*lengthscale'])

print('Feature ranking by length scale')
print(sorted_ls)
