import numpy
import tensorflow as tf

import layers as L

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.compat.v1.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.compat.v1.flags.DEFINE_boolean('top_bn', False, "")

temp = True

def logit(x, classes_count, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    global temp
    h = x
    if temp:
        temp = False
        print(type(x), x.shape)
    rng = numpy.random.RandomState(seed)
    layer_in, layer_out = 40, 64
    h = L.fc(h, layer_in, layer_out, seed=rng.randint(123456), name='fc 1')
    h = L.lrelu(L.bn(h, layer_out, is_training=is_training, update_batch_stats=update_batch_stats, name='layer 1'), FLAGS.lrelu_a)

    layer_in, layer_out = 64, 32
    h = L.fc(h, layer_in, layer_out, seed=rng.randint(123456), name='fc 2')
    h = L.lrelu(L.bn(h, layer_out, is_training=is_training, update_batch_stats=update_batch_stats, name='layer 2'), FLAGS.lrelu_a)

    layer_in, layer_out = 32, classes_count
    h = L.fc(h, layer_in, layer_out, seed=rng.randint(123456), name='fc 3')
    h = L.lrelu(L.bn(h, layer_out, is_training=is_training, update_batch_stats=update_batch_stats, name='layer 3'), FLAGS.lrelu_a)

    # if FLAGS.top_bn:
    #     h = L.bn(h, 10, is_training=is_training,
    #              update_batch_stats=update_batch_stats, name='bfc')

    return h
