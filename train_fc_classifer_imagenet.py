import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
#from imagenet_classes import class_names
from lib import *
import pdb

with open('imagenet-train.json') as fp:
    dataset=json.load(fp)
    fp.close()

with open('imagenet-val.json') as fp:
    test=json.load(fp)
    fp.close()

print(len(dataset['paths']))
batch_size=200


class vgg16:
    def __init__(self, imgs, labs, weights=None, sess=None):
        self.imgs = imgs
        self.labs = labs
        self.convlayers()
        #self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1),trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='features')

        print(self.pool5.name)

    # def fc_layers(self):
    #     # fc1
    #     with tf.name_scope('fc1') as scope:
    #         shape = int(np.prod(self.pool5.get_shape()[1:]))
    #         fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
    #                                                      dtype=tf.float32,
    #                                                      stddev=1e-1), name='weights')
    #         fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
    #                              trainable=True, name='biases')
    #         pool5_flat = tf.reshape(self.pool5, [-1, shape])
    #         fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    #         self.fc1 = tf.nn.relu(fc1l)
    #         self.parameters += [fc1w, fc1b]

    #     # fc2
    #     with tf.name_scope('fc2') as scope:
    #         fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
    #                                                      dtype=tf.float32,
    #                                                      stddev=1e-1), name='weights')
    #         fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
    #                              trainable=True, name='biases')
    #         fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
    #         self.fc2 = tf.nn.relu(fc2l)
    #         self.parameters += [fc2w, fc2b]

    #     # fc3
    #     with tf.name_scope('fc3') as scope:
    #         fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
    #                                                      dtype=tf.float32,
    #                                                      stddev=1e-1), name='weights')
    #         fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
    #                              trainable=True, name='biases')
    #         self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
    #         self.parameters += [fc3w, fc3b]

    #         print(self.fc3l.name)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print(type(keys))
        keys = [i for i in ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W',
 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b',
 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b']]
        print(type(keys))
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

def OneHot(l):
    x = np.zeros(1000)
    x[l] = 1.0
    return x


image_paths = tf.convert_to_tensor(dataset['paths'], dtype=tf.string)
labels = tf.convert_to_tensor(dataset['labels'], dtype=tf.int32)

test_image_paths = tf.convert_to_tensor(test['paths'], dtype=tf.string)
test_labels = tf.convert_to_tensor(test['labels'], dtype=tf.int32)


input_queue = tf.train.slice_input_producer([image_paths, labels], shuffle = True)
test_input_queue = tf.train.slice_input_producer([test_image_paths, test_labels], shuffle = True)

images = tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels = 3)
labels = input_queue[1]

test_images = tf.image.decode_jpeg(tf.read_file(test_input_queue[0]), channels = 3)
test_labels = test_input_queue[1]

images = tf.image.resize_images(images, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
test_images = tf.image.resize_images(test_images, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

images.set_shape([224, 224, 3])
test_images.set_shape([224, 224, 3])

#labels_batch = tf.one_hot(labels_batch, 1000)
images_batch, labels_batch = tf.train.batch([images, labels], batch_size = batch_size, num_threads = 2)
test_images_batch, test_labels_batch = tf.train.batch([test_images, test_labels], batch_size = batch_size, num_threads = 2)



if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_writer = tf.summary.FileWriter('logs/train', graph=sess.graph)
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3],name='imgs')
        labs = tf.placeholder(tf.float32, [None, 1000], name = 'labs')

        vgg = vgg16(imgs, labs, 'vgg16_weights.npz', sess)

        shape = int(np.prod(vgg.pool5.get_shape()[1:]))
        print('---------------------->',shape)
        final_features=np.zeros((0,shape))
        final_labels = np.zeros((0,1000))
        n_epoch=1

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        saver = tf.train.Saver()

        for e in range(n_epoch):

            print("Epoch "+str(e)+" #############################")

            for i in range(0, len(test['paths']), batch_size):
                print('batch ' ,i)
                path,  im, lab = sess.run([test_input_queue[0], test_images_batch, test_labels_batch])
                print('Here')

                one_hot_labels = np.zeros((batch_size,1000))
                for j in range(0,batch_size):
                    one_hot_labels[j] = OneHot(lab[j])

                feed_dict = {
                    'imgs:0': im,
                    'labs:0': one_hot_labels
                }

                a = sess.run([vgg.pool5], feed_dict = feed_dict)[0]
                features_flat = np.reshape(a, [-1, shape])
                
                saver.save(sess, 'logs/model', 1)
                train_writer.flush()
        print('SAVING....')

        coord.request_stop()
        coord.join(threads)
        sess.close()
