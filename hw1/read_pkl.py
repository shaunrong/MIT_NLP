#!/usr/bin/env python
import os

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


import cPickle

if __name__ == '__main__':
    results = {}
    for f in os.listdir('data'):
        if f[-3:] == 'pkl':
            temp_dict = cPickle.loads(open('data/{}'.format(f)).read())
            for key, value in temp_dict.iteritems():
                results[key] = value

    with open('results.pkl', 'w') as f:
        f.write(cPickle.dumps(results))

    best_LL = float('-inf')
    best_para = None

    for key, value in results.iteritems():
        if value[1] > best_LL:
            best_LL = value[1]
            best_para = {'n': key[0],
                         'd': key[1],
                         'm': key[2],
                         'iter': value[0]}

    print best_LL
    print best_para