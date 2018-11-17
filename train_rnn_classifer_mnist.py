import tensorflow as tf
import numpy as np
import cPickle
from tensorflow.python.ops import rnn, rnn_cell

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('B', 200, """Batch size""")
flags.DEFINE_integer('gpu', 0, """GPU no""")
flags.DEFINE_float('w_inv_loss', 100, """GPU no""")


from lib import *
import cifar10

def mnist():
    f = open('/mnist/mnist.pkl', 'rb')
    train, test, valid =  cPickle.load(f)
    train_ = {
        'data'  : train[0].reshape(-1, 28, 28,1),
        'labels': one_hot(train[1], 10)
    }
    test_ = {
        'data'  : test[0].reshape(-1, 28, 28,1),
        'labels': one_hot(test[1], 10)
    }
    valid_ = {
        'data'  : valid[0].reshape(-1, 28, 28,1),
        'labels': one_hot(valid[1], 10)
    }
    f.close()
    return train_, test_, valid_

train = cifar10.load(full= True)

# train, test, valid = mnist()
cursor = 0


def next_batch(nB):
    global cursor
    begin, end = None, None
    if cursor + nB < train.shape[0]:
        begin = cursor
    else:
        begin = cursor = 0
    end = cursor + nB
    cursor += nB
    return {
        'data'  : train['data'][begin:end],
        'labels': train['labels'][begin:end]
    }



def main(_):
	nH, nW, nD = 28, 28, 1
	nC = 10
	nB = FLAGS.B
	nT = 10
	
	with tf.device("/gpu:"+ str(FLAGS.gpu)):

            n_hidden = 10
            n_sequence = 4

            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev = 0.1)
                return tf.Variable(initial)
            
            def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)

            def RNN(x):
                x_list = []
                for i in range(n_sequence):
                    x_list.append(x)

                with tf.variable_scope('rnn'):
                    # Define a lstm cell with tensorflow
                    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden)

                    # Get lstm cell output
                    outputs, states = rnn.rnn(lstm_cell, x_list, dtype=tf.float32)

                    logits = []
                    for i in range(len(outputs)):
                        with tf.variable_scope('linear'+str(i)):
                            logits.append(linear(outputs[i], n_hidden, 2))
                    return logits

            def Hash(y):
                with tf.variable_scope('hash'):
                    y1 = relu(linear(y, 10, 100))
                    z_logits = []
                    for i in range(n_sequence):
                        with tf.variable_scope('linear'+str(i)):
                            z_logits.append(linear(y1, 100, 2))
                    return z_logits

            
            def InvHash(z_logits):
                with tf.variable_scope('invhash'):
                    z_concat = tf.concat(1, z_logits)
                    with tf.variable_scope('linear1'):
                        y1 = relu(linear(z_concat, 2*n_sequence, 100))
                    with tf.variable_scope('linear2'):
                        y__logit = linear(y1, 100, 10)
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
                v = np.zeros((10,10))
                for i in range(10):
                    v[i][i] = 1.0
                output = sess.run(z, feed_dict = {'y:0': v})
                s_list = []
                for i in range(10):
                    s = ''
                    for j in range(n_sequence):
                        if output[j][i][0] > output[j][i][1]:
                            s += '0'
                        else:
                            s += '1'
                    s_list.append(s)
                ok = True
                for i in range(10):
                    for j in range(i+1,10):
                        if s_list[i] == s_list[j]:
                            ok = False
                if ok:
                    print("Hash One to One")
                else:
                    print("Hash not one one", s_list)



            sess = create_session()
            saver = tf.train.import_meta_graph('saved/model-45000.meta')
            saver.restore(sess, tf.train.latest_checkpoint('saved/'))
            g = tf.get_default_graph()
            prev_vars = tf.all_variables()

            x = g.get_tensor_by_name('x:0')
            y = g.get_tensor_by_name('y:0')
            print x,y
            is_training = g.get_tensor_by_name('is_training:0')

            y_pred = g.get_tensor_by_name('cnn_train/y_pred_logits:0')
            y_pred = tf.stop_gradient(y_pred)

            features = g.get_tensor_by_name('cnn_train/features:0')
            features = tf.stop_gradient(features)
            accuracy = g.get_tensor_by_name('cnn_train/accuracy:0')

#---------------------------------------------------------------------------------------------#

            z_logits    = RNN(features)
            zlogits     = Hash(y)


            z_ = []
            for i in range(n_sequence):
                z_.append(tf.nn.softmax(z_logits[i]))

            z = []
            for i in range(n_sequence):
                z.append(softmax(zlogits[i]))

            y__logits = InvHash(z)

            

            eq_check_3 = cross_entropy(y__logits, y)
            tf.summary.scalar('inverse_loss', eq_check_3)

            l2 = 0
            for i in range(n_sequence):
                l2 += tf.nn.l2_loss(z[i])

            eq_check_2 = sum_cross_entropy(z_logits, z)
            tf.summary.scalar('rnn_classification_loss', eq_check_2)
            tf.summary.scalar('l2_z', l2)
            total_loss =  eq_check_2 + 100*eq_check_3 + 0.01*l2
            tf.summary.scalar('loss', total_loss)
            rnn_acc        = match_all(z_, z)
            tf.summary.scalar('rnn_acc', rnn_acc)


            learning_rate   = 0.0001

            with tf.variable_scope('optimizer'):
                optimizer= minimize(total_loss, { 'learning rate' : learning_rate}, algo='adam')

            sess.run(tf.initialize_variables(list(set(tf.all_variables()) - set(prev_vars)) ))

            writer = tf.summary.FileWriter('logs', graph = sess.graph)
            summary_op = tf.summary.merge_all()


            n_epoch         = 100
            n_batch         = 200
            
            n_display       = 10000
            
            for e in range(n_epoch):

                for i in range(0, train.shape[0], nB):

                    batch = next_batch(nB)

                    feed_dict = {
                        'x:0': batch['data'],
                        'y:0': batch['labels'],
                        'is_training:0': True
                    }
                    a = sess.run([optimizer,summary_op], feed_dict = feed_dict)
                    writer.add_summary(a[-1], e*50000 + i)

                    if i % 4000 == 0:
                        TestHash()
                        writer.flush()

                    
        


if __name__ == "__main__":
	tf.app.run()