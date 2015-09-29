#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

import numpy as np
import languagemodel as lm
import json
import cPickle


def train_nn(n, d, m):

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
    #n = 3    # Length of n-gram
    dim = d   # Word vector dimension
    hdim = m  # Hidden units
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

    return it, best_LL


def nearby(para_set):
    return [(min(para_set[0] + 1, 5), para_set[1], para_set[2],), (max(para_set[0] - 1, 2), para_set[1], para_set[2],),
            (para_set[0], para_set[1] + 1, para_set[2],), (para_set[0], para_set[1] - 1, para_set[2],),
            (para_set[0], para_set[1], para_set[2] + 1,), (para_set[0], para_set[1], para_set[2] - 1,)]


def optimize(para_set, results):
    feedback = True
    for para in nearby(para_set):
        if para not in results.keys():
            feedback = False
        else:
            if results[para] > results[para_set]:
                feedback = False
    return feedback


if __name__ == '__main__':
    results = cPickle.loads(open('results.pkl').read())
    current_para_set = (2, 10, 30,)
    print(current_para_set)
    if current_para_set not in results.keys():
        it, performance = train_nn(*current_para_set)
        results[current_para_set] = (it, performance, )
        with open('results.pkl', 'w') as f:
            f.write(cPickle.dumps(results))

    while not optimize(current_para_set, results):
        for para in nearby(current_para_set):
            print(para)
            results = cPickle.loads(open('results.pkl').read())
            if para not in results.keys():
                it, performance = train_nn(*para)
                results[para] = (it, performance, )
                with open('results.pkl', 'w') as f:
                    f.write(cPickle.dumps(results))

        best_para_set = current_para_set
        for para in nearby(current_para_set):
            if results[para][1] > results[best_para_set][1]:
                best_para_set = para

        current_para_set = best_para_set

    print("best para set is n = {}, d = {}, m = {}, it= {}, best performance is {}".format(current_para_set[0],
                                                                                    current_para_set[1],
                                                                                    current_para_set[2],
                                                                                    results[current_para_set][0],
                                                                                    results[current_para_set][1]))

    with open('para_space', 'w') as f:
        json.dump(results, f)