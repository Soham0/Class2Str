import numpy as np
import tensorflow as tf
import json
import cPickle as pickle
import matplotlib.pyplot as plt
import sys
import time
from    scipy.io    import loadmat
from    scipy.misc  import imread, imresize


# GPU Config #######################################################

gpu_memory              = 1.0
allow_growth            = True
allow_soft_placement    = True
log_device_placement    = False

# Function Defs ####################################################

args = sys.argv[1:]



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)



def sparsify(labels, max_time):
    indices = []
    values = []
    batch_size = len(labels)
    for b in range(batch_size):
        for t in range(len(labels[b])):
                # print b,t
            indices.append([b, t])
            values.append(labels[b][t])

    shape = [batch_size, max_time]
    return (np.array(indices, dtype=np.int64),
            np.array(values, dtype=np.int64),
            np.array(shape, dtype=np.int64))

def imshow(i):
    plt.imshow(i)
    plt.show()

def save(file_name, ob, format='numpy'):
    if format == 'json':
        with open(file_name, 'w') as outfile:
            json.dump(ob, outfile)
    elif format == 'numpy':
        np.save(file_name, ob)

def load(file_name, format='numpy'):

    

    if format == 'json':
        print("Loading file " + file_name)
        with open( file_name, 'r') as f:
            return json.load(f)
    elif format == 'file':
        print("Loading file " + file_name)
        with open( file_name, 'r') as f:
            return f.read()
    elif format == 'img':
        # print("Loading file " + file_name)
        return imread(file_name)
    elif format == 'numpy':
        print("Loading file " + file_name)
        return np.load(file_name + '.npy')
    elif format == 'npz':
        print("Loading file " + file_name)
        return np.load(file_name + '.npz')
    elif format == 'pickle':
        print("Loading file " + file_name)
        with open(file_name,'rb') as f:
            return pickle.load(f)
    elif format == 'mat':
        print("Loading file " + file_name)
        return loadmat(file_name + '.mat')

def weight(shape, init = None, trainable = True, decay = None, name='weights'):
    print(shape, trainable)
    initializer = None
    if init == None:
        initializer = tf.truncated_normal_initializer(stddev = 0.1)
    else:
        initializer = tf.constant_initializer(init)

    W = tf.get_variable(name, shape, initializer = initializer,
                dtype = tf.float32, trainable = trainable)
    if decay is not None:
        decay = tf.multiply(tf.nn.l2_loss(W), decay, name='weight_decay')
        tf.add_to_collection('regularizers', decay)
    return W
    

def bias(shape, init = 0.1,  trainable = True, name='bias'):
    print(shape, trainable)
    if init == None:
        init = 0.1
    return tf.get_variable(name, shape, initializer = tf.constant_initializer(init),
                dtype = tf.float32, trainable = trainable)




def get(d, key, default = None):
    if key in d:
        return d[key]
    else:
        return default
def Pass(x, network, is_training=True):
    y = x
    for i in range(len(network)):
        layer_type  = network[i][0]
        config      = {}

        if len(network[i]) == 2:
            config = network[i][1]

        name = get(config, 'name', layer_type + '_'+ str(i))
        print(name)

        trainable = get(config, 'trainable', True)

        with tf.variable_scope(name):
            if layer_type == 'conv':
                strides = get(config, 'strides', [1, 1, 1, 1])
                padding = get(config, 'padding', 'SAME')
                size    = get(config, 'size')
                W       = weight(size, get(config, 'W'), trainable, decay=1.0)
                b       = bias([size[3]], get(config, 'b', 0.1), trainable)

                act     = get(config, 'activation', 'relu')
                y       = tf.nn.conv2d(y, W, strides, padding=padding) + b
                bn      = get(config, 'batch norm', False)

                if bn:
                    y = tf.contrib.layers.batch_norm(y, is_training=is_training, scale=True, updates_collections=None, decay=0.9)
    
                if act == 'relu':
                    y   = tf.nn.relu(y)

            elif layer_type == 'normalize':
                mean    = get(config, 'mean', 0)
                std     = get(config, 'std', 1)
                y       = y - tf.constant(mean, dtype=tf.float32)
                y       = y / tf.constant(std, dtype=tf.float32)

            elif layer_type == 'pool':
                size    = get(config, 'size', [1, 2, 2, 1])
                strides = get(config, 'strides', [1, 2, 2, 1])
                padding = get(config, 'padding', 'VALID')
                y       = tf.nn.max_pool(y, ksize   = size, strides = strides,
                                                            padding = padding)

            elif layer_type == 'linear':
                size    = get(config, 'size')
                if len(size) == 1:
                    print y.get_shape()
                    size = [y.get_shape()[1]] + size
                W       = weight(size, get(config, 'W'), trainable, decay=1.)
                b       = bias([size[1]], get(config, 'b'), trainable)
                y       =  tf.matmul(y, W) + b
                activation = get(config, 'activation')
                if activation == 'relu':
                    y   = tf.nn.relu(y)

            elif layer_type == 'dropout':
                prob    = get(config, 'prob')
                y   = tf.nn.dropout(y, prob)

            elif layer_type == 'batch_norm':
                y = tf.contrib.layers.batch_norm(y, is_training=is_training, scale=True, updates_collections=None, decay=0.9)

            elif layer_type == 'relu':
                y       = tf.nn.relu(y)

            elif layer_type == 'reshape':
                size    = get(config, 'size', None)
                if size == None:
                    size = [-1, int(np.prod(y.get_shape()[1:]))]
                y       = tf.reshape(y, size)

            elif layer_type == 'softmax':
                d     = get(config, 'dim', -1)
                y     = y

            elif layer_type == 'lrn':
                radius  = get(config, 'radius', 2)
                alpha   = get(config, 'alpha', 2e-05)
                beta    = get(config, 'beta', 0.75)
                lrn_bias = get(config, 'bias', 1.0)
                y       = tf.nn.local_response_normalization(y,
                                                    depth_radius=radius,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=lrn_bias)


    return y

def image_read(file_name, H,W,D):

    c=(imread(file_name))
    if(len(c.shape)<3):
        c=np.dstack((c,c,c))

    c=(c[:,:,:3]).astype(np.uint8)
    c=np.dstack((c[:,:,2],c[:,:,1],c[:,:,0]))
    c=imresize(c, (H, W, D))

    return c


def load_batch(batch_paths, H, W, D):
    B = np.zeros([len(batch_paths), H, W, D])
    for i in range(len(batch_paths)):
        B[i] = image_read(batch_paths[i],H, W, D)
    return B

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
            
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def run(ops, feed, sess = None):
    if sess == None:
        sess = create_session()
    return sess.run(ops, feed_dict = feed)

def conv(x, config, strides =  [1, 1, 1, 1], padding = 'SAME', W = None, b = None):
    if W == None:
        W = weight_variable(config)
    if b == None:
        b = bias_variable([config[3]])
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides, padding ), b))

def pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1], padding = 'SAME')

def linear(x, n_input, n_output, W=None, b=None):
    if W == None:
        W = weight([n_input, n_output])
    if b == None:
        b = bias([n_output])
    return tf.matmul(x, W) + b

def relu(x):
    return tf.nn.relu(x)

# p is list of logits and q is a list of prob dist
def sum_cross_entropy(p,q):
    loss = 0
    for i in range(len(p)):
        loss += tf.nn.softmax_cross_entropy_with_logits(logits=p[i], labels=q[i])
    return tf.reduce_mean(loss)

# p is a logit and q is a prob dist
def cross_entropy(p,q):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=p, labels=q)
    return tf.reduce_mean(loss)

def create_session():
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction=gpu_memory # don't hog all vRAM
    config.gpu_options.allow_growth=allow_growth
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess

def reshape(x, shape):
    return tf.reshape(x, shape)

def match_all(y, y_):
    # Evaluate model
    num_correct_pred = 0
    for i in range(len(y)):
        num_correct_pred += tf.cast(tf.equal(tf.argmax(y_[i],1), tf.argmax(y
            [i],1)), tf.int32)
    correct_pred = tf.equal(num_correct_pred, tf.constant(len(y),dtype=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def match(y, y_):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1), tf.argmax(y,1)), tf.float32))
    return accuracy

def minimize(loss, config, algo="adam"):
    if algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate=config['learning rate']).minimize(loss)
    elif algo == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=config['learning rate']).minimize(loss)
    elif algo == 'momentum':
        return tf.train.MomentumOptimizer(config['learning rate'], config['momentum']).minimize(loss)


def one_hot(x, n_classes):

    l = len(x)

    y = np.zeros((l,n_classes))
    for i in range(l):
        z = np.zeros((n_classes))
        z[x[i]] = 1.0
        y[i] = z
    return y

def softmax(x):
    return tf.nn.softmax(x)
