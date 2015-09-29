#!/usr/bin/env python
import cPickle
import matplotlib.pyplot as plt

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'

if __name__ == '__main__':
    results = cPickle.loads(open('results.pkl').read())
    x = []
    y = []
    for key, value in results.iteritems():
        if key[0] == 2 and key[1] == 10:
            x.append(key[2])
            y.append(value[1])

    plt.style.use('ggplot')

    plt.plot(x, y, 'gs')
    plt.xlabel('m')
    plt.ylabel('log-likelihood')
    plt.title('log-likelihood v.s. m at n=2 and d=10')
    plt.show()


