import numpy as np
import cPickle

from lib import *

class MNIST(object):

    def __init__(self):
        f = open('/data4/girish.varma/mnist/mnist.pkl', 'rb')
        train, test, valid =  cPickle.load(f)
        self.train = {
            'data'  : train[0].reshape(-1, 28, 28,1),
            'labels': one_hot(train[1], 10)
        }
        self.test = {
            'data'  : test[0].reshape(-1, 28, 28,1),
            'labels': one_hot(test[1], 10)
        }
        self.valid = {
            'data'  : valid[0].reshape(-1, 28, 28,1),
            'labels': one_hot(valid[1], 10)
        }
        f.close()

        self.cursor = 0
        self.nC = 10
        self.nH, self.nW, self.nD = 28, 28, 1

    def dims(self):
        return self.nH, self.nW, self.nD


    def next_batch(self, nB, phase = 'train'):
        if phase == 'train':
            begin, end = None, None
            if self.cursor + nB < 50000:
                begin = self.cursor
            else:
                begin = self.cursor = 0
            end = self.cursor + nB
            self.cursor += nB
            return {
                'data'  : self.train['data'][begin:end],
                'labels': self.train['labels'][begin:end]
            }
        else:
            return {
                'data'  : self.test['data'][:nB],
                'labels': self.test['labels'][:nB]
            }


    def test_batch(self, nT, phase = 'test'):
        
        begin, end = None, None
        begin = 0
        end = nT
        return {
            'data'  : self.test['data'][begin:end],
            'labels': self.test['labels'][begin:end]
        }


