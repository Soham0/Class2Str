from __future__ import print_function


import numpy as np
# import matplotlib.pylab as plt


import cPickle



class CIFAR100(object):

    def __init__(self, full = True, flatten = False):
        
        cifar100 = {'data': [], 'labels':[]}
        f = open('/data4/girish.varma/cifar100/train', 'rb')
        c = cPickle.load(f)
        f.close()

        cifar100['data'] = c['data']
        cifar100['labels'] = c['fine_labels']
        
        n = cifar100['data'].shape[0]

        if flatten:
            cifar100['data'] = cifar100['data']
        else:
            cifar100['data'] = cifar100['data'].reshape((n, 3, 32, 32)).transpose(0, 2, 3, 1)
        
        labels = np.zeros((n,100))

        for i in range(n):
            labels[i] = OneHot(cifar100['labels'][i])
        cifar100['one_hot_labels'] = labels
        

        #### Test Data ####

        cifar100_test = {'data':[],'labels':[]}
        f = open('/data4/girish.varma/cifar100/test', 'rb')
        c = cPickle.load(f)
        f.close()

        cifar100_test['data']=c['data']
        cifar100_test['labels']=c['fine_labels']

        n = cifar100_test['data'].shape[0]

        if flatten:
            cifar100_test['data'] = cifar100_test['data']
        else:
            cifar100_test['data'] = cifar100_test['data'].reshape((n, 3, 32, 32)).transpose(0, 2, 3, 1)


        labels = np.zeros((n,100))

        for i in range(n):
            labels[i] = OneHot(cifar100_test['labels'][i])
        
        cifar100_test['one_hot_labels'] = labels

        self.train =  cifar100
        self.test  =  cifar100_test
        self.cursor = 0
        self.test_cursor = 0
        self.nC = 100
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
    x = np.zeros(100)
    x[l] = 1.0
    return x
