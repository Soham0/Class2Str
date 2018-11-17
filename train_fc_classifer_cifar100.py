import tensorflow as tf
import numpy as np
import timeit

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('B', 200, """Batch size""")
flags.DEFINE_integer('F', 64, """Initial filter size for CNN""")
flags.DEFINE_string('dataset', 'cifar100', """Dataset""")
flags.DEFINE_string('name', 'default', """Name of run""")
flags.DEFINE_string('model', 'vgg16', """Name of run""")

from lib import *

def init():
	if FLAGS.dataset == 'mnist':
		import mnist
		dataset = mnist.MNIST()
	elif FLAGS.dataset == 'cifar10':
		import cifar10
		dataset = cifar10.CIFAR10()

	elif FLAGS.dataset == 'cifar100':
		import cifar100
		dataset = cifar100.CIFAR100()

	nH, nW, nD = dataset.dims()
	nC = dataset.nC
	print('-------------------------------------------',nC)
	
	if FLAGS.model == 'vgg16':
		import vgg16
		model = vgg16.VGG16(nH, nW, nD, FLAGS.F, nC)

	return dataset, model





def main(_):
	dataset, model = init()

	nB = FLAGS.B
	
	nT = 10
	

	with tf.variable_scope('cnn_train'):
		y_pred = softmax(model.logits)
		loss_classifier = cross_entropy(model.logits, model.y)
		
		regularizer = tf.add_n(tf.get_collection('regularizers'))
		loss = loss_classifier + 0.001*regularizer
		loss=tf.identity(loss,name='loss')
		optimizer = minimize(loss, { 'learning rate' : 0.0001}, algo='adam')
		# print(type(optimizer))
		acc = tf.identity(match(y_pred, model.y), name='accuracy')
		tf.summary.scalar('loss_classifier', loss_classifier)
		tf.summary.scalar('regularizer', regularizer)
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('acc', acc)

		summ_op = tf.summary.merge_all()
		saver = tf.train.Saver()
		sess = create_session()

		train_writer = tf.summary.FileWriter(FLAGS.name + '/logs/train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.name + '/logs/test')

		print(model.x.name)
		print(model.y.name)
		print(model.is_training.name)
		print(model.dr_p1.name)
		print(model.dr_p2.name)
		print(model.dr_p3.name)
		print(model.features.name)
		print(model.logits.name)
		print(acc.name)

		nE = 1000

		for e in range(nE):
			print('************* EPOCH ' + str(e) + ' *************')
			for i in range(0, dataset.train['data'].shape[0], nB):
				batch = dataset.next_batch(nB)
				a = model.train(sess, optimizer, summ_op, batch)
				train_writer.add_summary(a, e*dataset.train['data'].shape[0] + i)

				if i % (nB*10) == 0:
					batch = dataset.test_batch(nB)
					start = timeit.default_timer()
					a = model.test(sess, summ_op, batch)
					stop = timeit.default_timer()
					print('----------------------->>>>',(float(start-stop))/nB)
					test_writer.add_summary(a, e*dataset.train['data'].shape[0] + i)
					train_writer.flush()
					test_writer.flush()
					print i




				if i % 10000 == 0:
					saver.save(sess, FLAGS.name + '/model', i)
					train_writer.flush()
					test_writer.flush()


if __name__ == "__main__":
	tf.app.run()