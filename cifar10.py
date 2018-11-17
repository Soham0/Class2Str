from __future__ import print_function


import numpy as np

import cPickle



class CIFAR10(object):

    def __init__(self, full = True, flatten = False):
        cifar10 = {'data': [], 'labels':[]}
        j = 2
        if full:
            j = 6
        for i in range(1,j):
            f = open('/cifar10/data_batch_'+ str(i), 'rb')
            c = cPickle.load(f)
            f.close()
            if i == 1:
                cifar10['data'] = c['data']
            else:
                cifar10['data'] = np.vstack([cifar10['data'], c['data']])
            cifar10['labels'] += c['labels']

        n = cifar10['data'].shape[0]
        if flatten:
            cifar10['data'] = cifar10['data']
        else:
            cifar10['data'] = cifar10['data'].reshape((n, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.zeros((n,10))
        for i in range(n):
            labels[i] = OneHot(cifar10['labels'][i])
        cifar10['one_hot_labels'] = labels
        

        #### Test Data ####

        cifar10_test = {'data':[],'labels':[]}
        f = open('/data4/girish.varma/cifar10/test_batch', 'rb')
        c = cPickle.load(f)
        f.close()

        cifar10_test['data']=c['data']
        cifar10_test['labels']=c['labels']
        n = cifar10_test['data'].shape[0]

        if flatten:
            cifar10_test['data'] = cifar10_test['data']
        else:
            cifar10_test['data'] = cifar10_test['data'].reshape((n, 3, 32, 32)).transpose(0, 2, 3, 1)


        labels = np.zeros((n,10))
        for i in range(n):
            labels[i] = OneHot(cifar10_test['labels'][i])
        cifar10_test['one_hot_labels'] = labels

        self.train =  cifar10
        self.test  =  cifar10_test
        self.cursor = 0
        self.test_cursor = 0
        self.nC = 10
        self.nH, self.nW, self.nD = 32, 32, 3



    def dims(self):
        return self.nH, self.nW, self.nD


    def next_batch(self, nB, phase = 'train'):
        
        begin, end = None, None
        if self.cursor + nB < 50000:
            begin = self.cursor
        else:
            begin = self.cursor = 0
        end = self.cursor + nB
        self.cursor += nB
        return {
            'data'  : self.train['data'][begin:end],
            'labels': self.train['one_hot_labels'][begin:end]
        }


    def test_batch(self, nB, phase = 'test'):
        
        begin, end = None, None
        
        if self.test_cursor + nB < 10000:
            begin = self.test_cursor
        else:
            begin = self.test_cursor = 0
        end = self.test_cursor + nB
        self.test_cursor += nB
        
        return {
            'data'  : self.test['data'][begin:end],
            'labels': self.test['one_hot_labels'][begin:end]
        }
        




def OneHot(l):
    x = np.zeros(10)
    x[l] = 1.0
    return x
