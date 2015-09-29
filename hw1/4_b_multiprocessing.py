#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import itertools

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np
import languagemodel as lm
import multiprocessing
import cPickle


def train_nn(params):
    print(params)

    np.random.seed(1)  # for reproducibility

    corpus_train = lm.readCorpus("data/train.txt")
    corpus_dev = lm.readCorpus("data/dev.txt")
    corpus_test = lm.readCorpus("data/test.txt")

    # build a common index (words to integers), mapping rare words (less than 5 occurences) to index 0
    # nwords = vocabulary size for the models that only see the indexes

    w2index, nwords = lm.buildIndex(corpus_train + corpus_dev + corpus_test)

    # find words that appear in the training set so we can deal with new words separately
    count_train = np.zeros((nwords,))
    for snt in corpus_train:
        for w in snt:
            count_train[w2index[w]] += 1

    # Network model
    #print("\nNetwork model training:")
    n = params[0]    # Length of n-gram
    dim = params[1]   # Word vector dimension
    hdim = params[2]  # Hidden units
    neurallm = lm.neuralLM(dim, n, hdim, nwords)  # The network model

    ngrams = lm.ngramGen(corpus_train, w2index, n)
    ngrams2 = lm.ngramGen(corpus_dev, w2index, n)

    lrate = 0.5  # Learning rate
    best_LL = float('-inf')
    it = 0
    while True:
        it += 1
        LL, N = 0.0, 0  # Average log-likelihood, number of ngrams
        for ng in ngrams:
            pr = neurallm.update(ng, lrate)
            LL += np.log(pr)
            N += 1
        #print('Train:\t{0}\tLL = {1}'.format(it, LL / N))

        #Dev set
        LL, N = 0.0, 0 # Average log-likelihood, number of ngrams
        for ng in ngrams2:
            if (count_train[ng[-1]]>0): # for now, skip target words not seen in training
                pr = neurallm.prob(ng)
                LL += np.log(pr)
                N += 1

        if LL / N > best_LL:
            best_LL = LL / N
        else:
            break

    return_result = {(params[0], params[1], params[2],): (it, best_LL)}

    with open('data/{}_{}_{}.pkl'.format(params[0], params[1], params[2]), 'w') as f:
        f.write(cPickle.dumps(return_result))


if __name__ == '__main__':
    para_n = [2, 3, 4, 5]
    para_d = range(5, 16)
    para_m = range(20, 41, 2)
    params = []
    for para in itertools.product(para_n, para_d, para_m):
        params.append(para)

    pool = multiprocessing.Pool()
    pool.map(train_nn, params)