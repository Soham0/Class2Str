import tensorflow as tf
import numpy as np

from lib import *

class LENET:

    

    def __init__(self, nH, nW, nD, nF, nC):
    
        self.x = tf.placeholder(tf.float32, [None, nH, nW, nD], name = 'x')
        self.y = tf.placeholder(tf.float32, [None, nC], name = 'y')
        self.is_training = tf.placeholder(tf.bool,[])


        cnn = [
			['conv', {
				'size': [5, 5, self.x.get_shape()[3].value, 8]
			}],
			['pool'],
			['conv', {
				'size': [3, 3, 8, 16],
				'batch norm': True
			}],
			['pool'],
			['conv', {
				'size': [3, 3, 16, 16],
				'batch norm': True
			}],
			['reshape']
		]

        classifier = [
			['linear', {
				'size': [300]
			}],
            ['linear', {
                'size': [100]
            }],
			['linear', {
				'size': [nC]
			}]
		]

        with tf.variable_scope('cnn'):
            self.features = tf.identity(Pass(self.x, cnn, self.is_training), name = 'features')

        with tf.variable_scope('classifier'):
            self.logits = tf.identity(Pass(self.features, classifier, self.is_training), name='logits')


    

    def train(self, sess, optimizer, summary, batch):

        feed_dict = {
            self.x: batch['data'],
            self.y: batch['labels'],
            self.is_training: True
        }

        a = sess.run([optimizer, summary], feed_dict)
        return a[-1]


    def test(self, sess, summary, batch):
        feed_dict = {
            self.x: batch['data'],
            self.y: batch['labels'],
            self.is_training: False
        }

        a = sess.run(summary, feed_dict)
        return a


