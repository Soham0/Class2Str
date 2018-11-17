import tensorflow as tf

import json
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn
from lib import *
import cPickle

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('B', 100, """Batch size""")
flags.DEFINE_integer('F', 64, """Initial filter size for CNN""")
flags.DEFINE_string('dataset', 'imagenet', """Dataset""")
flags.DEFINE_string('name', 'default', """Name of run""")
flags.DEFINE_string('model', 'vgg', """Name of run""")
flags.DEFINE_integer('restore', 0, """Binary variable to indicate whether to resume training or not""")

flags.DEFINE_float('w_c',1.,"Weight for classification"  )
flags.DEFINE_float('w_i', 1., "Weight for inverse")
flags.DEFINE_float('w_l2', 1., "Weight for l2")


batch_size = FLAGS.B

nB = FLAGS.B


def OneHot(l):
    x = np.zeros(1000)
    x[l] = 1.0
    return x

with open('imagenet-train.json') as fp:
    dataset=json.load(fp)
    fp.close()

with open('imagenet-val.json') as fp:
    test=json.load(fp)
    fp.close()

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


print(len(dataset['paths']))


#----------------------------------------------------------------------------------------------------#



image_paths = tf.convert_to_tensor(dataset['paths'], dtype=tf.string)
labels = tf.convert_to_tensor(dataset['labels'], dtype=tf.int32)

test_image_paths = tf.convert_to_tensor(test['paths'], dtype=tf.string)
test_labels = tf.convert_to_tensor(test['labels'], dtype=tf.int32)


input_queue = tf.train.slice_input_producer([image_paths, labels], shuffle = True, name='rnn_train_input_producer')
test_input_queue = tf.train.slice_input_producer([test_image_paths, test_labels], shuffle = True, name='rnn_test_input_producer')

images = tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels = 3)
labels = input_queue[1]

test_images = tf.image.decode_jpeg(tf.read_file(test_input_queue[0]), channels = 3)
test_labels = test_input_queue[1]

images = tf.image.resize_images(images, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
test_images = tf.image.resize_images(test_images, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

images.set_shape([224, 224, 3])
test_images.set_shape([224, 224, 3])

images_batch, labels_batch = tf.train.batch([images, labels], batch_size = batch_size, num_threads = 4, name='rnn_train_batch')
test_images_batch, test_labels_batch = tf.train.batch([test_images, test_labels], batch_size = batch_size, num_threads = 4,name='rnn_test_batch')

##-----------------------------------------------------------------------------------------------------------------------###########

def main(_):
    n_hidden=2000
    n_sequence=11
    nC = 1000
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
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_list, dtype=tf.float32, time_major = True, sequence_length = tf.constant([n_sequence]*batch_size))
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
            y1 = relu(linear(y, nC, 2000))
            z_logits = []
            for i in range(n_sequence):
                with tf.variable_scope('linear'+str(i)):
                    z_logits.append(linear(y1, 2000, 2))
            return z_logits

    
    def InvHash(z_logits):
         with tf.variable_scope('invhash'):
            print('-------------->',len(z_logits))

            for i in z_logits:
                print(i.get_shape())
            z_concat = tf.concat(z_logits,axis= 1)
            print('-------------->',z_concat.get_shape())

            with tf.variable_scope('linear1'):
                y1 = relu(linear(z_concat, 2*n_sequence, 2000))
            with tf.variable_scope('linear2'):
                y__logit = linear(y1, 2000, nC)
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
        v = np.zeros((1000,1000))
        for i in range(1000):
            v[i][i] = 1.0
        output = sess.run(z, feed_dict = {'labs:0': v})
        s_list = []
        for i in range(1000):
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
        for i in range(1000):
            for j in range(i+1,1000):
                if s_list[i] == s_list[j]:
                    ok = False
        if ok:
            print("Hash One to One")
            with open('imagenet1000_strings.pkl', 'wb') as fp:
                cPickle.dump(s_list, fp)
        else:
            print("Hash not one one", num_collisions)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        saver = tf.train.import_meta_graph('logs/model-1.meta')
        saver.restore(sess, tf.train.latest_checkpoint('logs/'))

        g = tf.get_default_graph()
        prev_vars = tf.all_variables()

        x = g.get_tensor_by_name('imgs:0')
        y = g.get_tensor_by_name('labs:0')

        print x,y

        features = g.get_tensor_by_name('features:0')
        features = tf.stop_gradient(features)

        features = tf.reshape(features, [-1, 7*7*512])

        #---------------------------------------------------------------------#

        x_list = []
        for i in range(n_sequence):
            x_list.append(features)

        x_list = tf.stack(x_list, axis = 0)

        z_logits    = rnn_layer(x_list,n_hidden,name='lstm1')
        z_logits    = rnn_layer(z_logits,n_hidden,name='lstm2')
        z_logits    = seq_linear(z_logits,'seq_linear1')        

        zlogits     = Hash(y)


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

        rnn_classification_loss = sum_cross_entropy(z_logits, z)
        tf.summary.scalar('rnn_classification_loss', rnn_classification_loss)
        tf.summary.scalar('l2_z', l2)
        total_loss =  FLAGS.w_c*rnn_classification_loss + FLAGS.w_i*inverse_loss + FLAGS.w_l2*l2
        tf.summary.scalar('loss', total_loss)
        rnn_acc        = match_all(z_, z)
        tf.summary.scalar('rnn_acc', rnn_acc)


        learning_rate   = 0.0001

        with tf.variable_scope('optimizer'):
            optimizer= minimize(total_loss, { 'learning rate' : learning_rate}, algo='adam')

        print(optimizer.name)

        sess.run(tf.initialize_variables(list(set(tf.all_variables()) - set(prev_vars)) ))


        train_writer = tf.summary.FileWriter('rnnlogs1/train', graph=sess.graph)
        test_writer = tf.summary.FileWriter('rnnlogs1/test')
        summary_op = tf.summary.merge_all()


        n_epoch         = 500
        n_batch         = 200
        n_display       = 10000


    #----------------------------------------------------------------------------------------------#

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

        if FLAGS.restore:
            s = tf.train.Saver()
            s.restore(sess, tf.train.latest_checkpoint('rnnlogs1/'))

        saver2 = tf.train.Saver()

        for e in range(n_epoch):

            print("Epoch "+str(e)+" #############################")

            for i in range(0, len(dataset['paths']), batch_size):
                path,  im, lab = sess.run([input_queue[0], images_batch, labels_batch])

                one_hot_labels = np.zeros((batch_size,1000))
                for j in range(0,batch_size):
                    one_hot_labels[j] = OneHot(lab[j])

                feed_dict = {
                    'imgs:0': im,
                    'labs:0': one_hot_labels
                }

                a = sess.run([optimizer,summary_op] + z, feed_dict = feed_dict)
                print find_unbiased(a[2:])

                train_writer.add_summary(a[1], e*len(dataset['paths']) + i)

                if i % 50000 == 0:

                    TestHash()

                    test_path,  test_im, test_lab = sess.run([test_input_queue[0], test_images_batch, test_labels_batch])

                    one_hot_labels = np.zeros((batch_size,1000))

                    for j in range(0,batch_size):
                        one_hot_labels[j] = OneHot(test_lab[j])

                    feed_dict = {
                        'imgs:0': test_im,
                        'labs:0': one_hot_labels
                    }

                    a = sess.run([summary_op], feed_dict = feed_dict)
                    test_writer.add_summary(a[0], e*len(dataset['paths']) + i)

                    saver2.save(sess,'rnnlogs1/model', i)
                    train_writer.flush()
                    test_writer.flush()



        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == "__main__":
    tf.app.run()
