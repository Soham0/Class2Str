import tensorflow as tf
import numpy as np
import cPickle
from tensorflow.python.ops import rnn
import timeit

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('B', 200, """Batch size""")
flags.DEFINE_integer('F', 64, """Initial filter size for CNN""")
flags.DEFINE_string('dataset', 'cifar10', """Dataset""")
flags.DEFINE_string('name', 'default', """Name of run""")
flags.DEFINE_string('model', 'vgg16', """Name of run""")

flags.DEFINE_float('w_c',1.,"Weight for classification"  )
flags.DEFINE_float('w_i', 1., "Weight for inverse")
flags.DEFINE_float('w_l2', 1., "Weight for l2")

from lib import *

def init():
    if FLAGS.dataset == 'mnist':
        import mnist
        dataset = mnist.MNIST()
    elif FLAGS.dataset == 'cifar10':
        import cifar10_synthetic
        dataset = cifar10_synthetic.CIFAR10()

    nH, nW, nD = dataset.dims()
    nC = dataset.nC
    
    if FLAGS.model == 'vgg16':
        import vgg16
        
    return dataset

def find_unbiased(a):
    unbiased = 1.0
    for i in range(len(a)):
        for j in range(a[i].shape[0]):
            p,q = 0.,0.
            if a[i][j,0] >= a[i][j,1]:
                p = a[i][j,0]
                q = a[i][j,1]
            else:
                p = a[i][j,1]
                q = a[i][j,0]
            if p < unbiased:
                unbiased = p
    return p


def main(_):
    dataset = init()
    n_hidden=100
    n_sequence=4
	nC = 10
    nB = FLAGS.B
    nT = 10


    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def rnn_layer(x,n_hidden,name='rnn'):
        with tf.variable_scope(name):
            n_sequence = x.get_shape().as_list()[0]
            lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, time_major = True, sequence_length = tf.constant([n_sequence]*nB))
            return outputs

    def RNN(x):
        print x.get_shape().as_list()
        x_list = []
        for i in range(n_sequence):
            x_list.append(x)

        x_list = tf.stack(x_list, axis = 0)

        print x_list.get_shape().as_list()

        with tf.variable_scope('rnn'):
            # Define a lstm cell with tensorflow
            lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_list, dtype=tf.float32, time_major = True, sequence_length = tf.constant([n_sequence]*nB))
            logits = []

            for i in range(n_sequence):
                with tf.variable_scope('linear'+str(i)):
                    logits.append(linear(outputs[i], n_hidden, 2))
            return logits

    def seq_linear(x,name='seq_linear'):
        logits = []
        n_sequence,_,n_hidden = x.get_shape().as_list()

        with tf.variable_scope(name):
            for i in range(n_sequence):
                with tf.variable_scope('linear'+str(i)):
                    logits.append(linear(x[i], n_hidden, 2))
            return logits

    def Hash(y):
        with tf.variable_scope('hash'):
            y1 = relu(linear(y, nC, 500))
            z_logits = []
            for i in range(n_sequence):
                with tf.variable_scope('linear'+str(i)):
                    z_logits.append(linear(y1, 500, 2))
            return z_logits

    
    def InvHash(z_logits):
         with tf.variable_scope('invhash'):
            print('-------------->',len(z_logits))

            for i in z_logits:
                print(i.get_shape())
            z_concat = tf.concat(z_logits,axis= 1)
            print('-------------->',z_concat.get_shape())

            with tf.variable_scope('linear1'):
                y1 = relu(linear(z_concat, 2*n_sequence, 500))
            with tf.variable_scope('linear2'):
                y__logit = linear(y1, 500, nC)
            return y__logit

    def match_all(y, y_):
        # Evaluate model
        num_correct_pred = 0
        for i in range(n_sequence):
            num_correct_pred += tf.cast(tf.equal(tf.argmax(y_[i],1), tf.argmax(y[i],1)), tf.int32)
        correct_pred = tf.equal(num_correct_pred, tf.constant(n_sequence,dtype=tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy



    def TestHash():
        C = 10
        v = np.zeros((C, C))
        for i in range(C):
            v[i][i] = 1.0
        output = sess.run(z, feed_dict = {'y:0': v})
        s_list = []
        for i in range(C):
            s = ''
            for j in range(n_sequence):
                if output[j][i][0] > output[j][i][1]:
                    s += '0'
                else:
                    s += '1'
            s_list.append(s)
        freq = {}
        num_collisions = 0
        for s in s_list:
            if s in freq.keys():
                freq[s] += 1
                num_collisions += 1
            else:
                freq[s] = 1



        ok = True
        for i in range(C):
            for j in range(i+1,C):
                if s_list[i] == s_list[j]:
                    ok = False
        if ok:
            print("Hash One to One")
            with open('cifar10_strings.pkl', 'wb') as fp:
                cPickle.dump(s_list, fp)
        else:
            print("Hash not one one", num_collisions)


    def rev_one_hot(x, y):
        n = x.shape[0]
        temp_labelsx=[]
        temp_labelsy=[]
        for i in range(n):
            temp_labelsx.append(np.argmax(x[i]))
            temp_labelsy.append(np.argmax(y[i]))
        x= temp_labelsx
        y= temp_labelsy
        return x,y

    def labels_concatenate(x,y):
        n = len(x)
        temp_labels=[]
        for i in range(n):
            temp_labels.append(x[i]*10 + y[i])

        return temp_labels

    sess = create_session()
    saver = tf.train.import_meta_graph('cifar10_cnn/model-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('cifar10_cnn/'))
    g = tf.get_default_graph()
    prev_vars = tf.all_variables()

    x = g.get_tensor_by_name('x:0')
    y = g.get_tensor_by_name('y:0')
    print x,y
    
    y_pred = g.get_tensor_by_name('classifier/logits:0')
    y_pred = tf.stop_gradient(y_pred)

    features = g.get_tensor_by_name('cnn/features:0')
    features = tf.stop_gradient(features)
    accuracy = g.get_tensor_by_name('cnn_train/accuracy:0')
    tf.summary.scalar('cnn_accuracy',accuracy)
#---------------------------------------------------------------------------------------------#

    x_list = []
    for i in range(n_sequence):
        x_list.append(features)

    x_list = tf.stack(x_list, axis = 0)

    z_logits    = rnn_layer(x_list,n_hidden,name='lstm1')
    z_logits    = rnn_layer(z_logits,20,name='lstm2')
    z_logits    = rnn_layer(z_logits,n_hidden,name='lstm3')
    z_logits    = seq_linear(z_logits,'seq_linear1')


    zlogits     = Hash(y)

    for i in range(0,len(z_logits)):
        print(z_logits[i].name)

    for i in range(0,len(zlogits)):
        print(zlogits[i].name)


    z_ = []
    for i in range(n_sequence):
        z_.append(tf.nn.softmax(z_logits[i]))

    z = []
    for i in range(n_sequence):
        z.append(softmax(zlogits[i]))

    y__logits = InvHash(z)

    

    inverse_loss = cross_entropy(y__logits, y)
    tf.summary.scalar('inverse_loss', inverse_loss)
    tf.summary.scalar('inverse_acc', match(y__logits, y))

    l2 = 0
    for i in range(n_sequence):
        l2 += tf.nn.l2_loss(z[i])

    l2 = -l2

    rnn_classification_loss = weighted_sum_cross_entropy(z_logits, z)
    tf.summary.scalar('rnn_classification_loss', rnn_classification_loss)
    tf.summary.scalar('l2_z', l2)

    total_loss =  FLAGS.w_c*rnn_classification_loss + FLAGS.w_i*inverse_loss + FLAGS.w_l2*l2
    tf.summary.scalar('loss', total_loss)
    rnn_acc        = match_all(z_, z)
    tf.summary.scalar('rnn_acc', rnn_acc)


    learning_rate   = 0.0001

    with tf.variable_scope('optimizer'):
        optimizer= minimize(total_loss, { 'learning rate' : learning_rate}, algo='adam')

    sess.run(tf.initialize_variables(list(set(tf.all_variables()) - set(prev_vars)) ))

    train_writer = tf.summary.FileWriter('cifar10_rnn/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter('cifar10_rnn/test')
    summary_op = tf.summary.merge_all()


    n_epoch         = 1000
    n_batch         = 200
    
    n_display       = 10000

    
    saver2 = tf.train.Saver()

    

    for e in range(n_epoch):

        print("Epoch "+str(e)+" #############################")

        for i in range(0, dataset.train['data'].shape[0], nB):

            batch = dataset.next_batch(nB)

            feed_dict = {
                'x:0': batch['data'],
                'y:0': batch['labels'],
                'Placeholder:0':False,
                'Placeholder_1:0':1.,
                'Placeholder_2:0':1.,
                'Placeholder_3:0':1.
            }

            # print('-----------------------------------------------------------------------------------------------')
            a = sess.run([optimizer,summary_op] + z , feed_dict = feed_dict)
            # print('RNN accuracy is: '+ str(a[-1]))

            print find_unbiased(a[2:])
            train_writer.add_summary(a[1], e*50000 + i)

            if i % 10000 == 0:
                TestHash()
                batch = dataset.test_batch(nB)

                feed_dict = {
                'x:0': batch['data'],
                'y:0': batch['labels'],
                'Placeholder:0':False,
                'Placeholder_1:0':1.,
                'Placeholder_2:0':1.,
                'Placeholder_3:0':1.
                }


                # print('-----------------------------------------------------------------------------------------------')
                start = timeit.default_timer()
                a = sess.run([summary_op] + z, feed_dict = feed_dict)
                stop = timeit.default_timer()

                print('----------------------->>>>',(float(start-stop))/nB)

                test_writer.add_summary(a[0], e*50000 + i)

                saver2.save(sess,'cifar10_rnn/model', i)
                train_writer.flush()
                test_writer.flush()

                    
        


if __name__ == "__main__":
	tf.app.run()
