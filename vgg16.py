import tensorflow as tf
import numpy as np

from lib import *
import timeit

class VGG16:

    

    def __init__(self, nH, nW, nD, nF, nC):
    
        self.x = tf.placeholder(tf.float32, [None, nH, nW, nD], name = 'x')
        self.y = tf.placeholder(tf.float32, [None, nC], name = 'y')
        self.is_training = tf.placeholder(tf.bool,[])
        self.dr_p1 = tf.placeholder(tf.float32, [])
        self.dr_p2 = tf.placeholder(tf.float32, [])
        self.dr_p3 = tf.placeholder(tf.float32, [])

        def ConvBNReLU(nI, nO):
            return ['conv', {
                'size'      : [3, 3, nI, nO],
                'activation': 'relu',
                'batch norm': True
            }]


        cnn = [
            
            ConvBNReLU(self.x.get_shape()[3].value, nF)
            ,
            ['dropout', {
                'prob'       : self.dr_p1
            }],
            ConvBNReLU(nF, nF)
            ,
            ['pool'],

            ConvBNReLU(nF, 2*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(2*nF, 2*nF)
            ,
            ['pool'],

            ConvBNReLU(2*nF, 4*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(4*nF, 4*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(4*nF, 4*nF),
            ['pool'],

            ConvBNReLU(4*nF, 8*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(8*nF, 8*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(8*nF, 8*nF),
            #['pool'],

            ConvBNReLU(8*nF, 8*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(8*nF, 8*nF),
            ['dropout', {
                'prob'       : self.dr_p2
            }],
            ConvBNReLU(8*nF, 8*nF),

            ['reshape']
        ]

        classifier = [
            ['dropout', {
                'prob': self.dr_p3
            }],
			['linear', {
				'size': [1024]
			}],
            ['batch_norm'],
            ['relu'],
            ['dropout', {
                'prob': self.dr_p3
            }],
            ['linear', {
                'size': [1024]
            }],
            ['batch_norm'],
            ['relu'],
            ['dropout', {
                'prob': self.dr_p3
            }],
			['linear', {
				'size': [nC]
			}]
		]

        with tf.variable_scope('cnn'):
            self.features = tf.identity(Pass(self.x, cnn, self.is_training), name = 'features')

        start = timeit.default_timer()
        with tf.variable_scope('classifier'):
            self.logits = tf.identity(Pass(self.features, classifier, self.is_training), name='logits')
        stop = timeit.default_timer()

        self.time = float(stop-start)


    

    def train(self, sess, optimizer, summary, batch):

        feed_dict = {
            self.x: batch['data'],
            self.y: batch['labels'],
            self.is_training: True,
            self.dr_p1 : 0.7,
            self.dr_p2 : 0.6,
            self.dr_p3 : 0.5
        }

        a = sess.run([optimizer, summary], feed_dict)
        return a[-1]


    def test(self, sess, summary, batch):
        feed_dict = {
            self.x: batch['data'],
            self.y: batch['labels'],
            self.is_training: False,
            self.dr_p1 : 1.,
            self.dr_p2 : 1.,
            self.dr_p3 : 1.
        }

        a = sess.run(summary, feed_dict)
        return a


