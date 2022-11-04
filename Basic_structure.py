import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from Utlis2 import *

d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')
'''
e_bn2 = batch_norm(name='e_bn2')
e_bn3 = batch_norm(name='e_bn3')
e_bn4 = batch_norm(name='e_bn4')
'''
g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')



def Encoder_SVHN_Specific(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h5 = linear(image, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean,z_log_sigma_sq,h5

def Encoder_SVHN_Shared(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        size = 2
        z_dim = 256
        h1 = conv2d(image, 64*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        return h4

def Generator_SVHN(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        #h8 = tf.nn.tanh(h8)
        h8 = tf.nn.sigmoid(h8)
        return h8

def Generator_SVHN_TeacherStudent(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3

        count = 2
        # fully-connected layers
        h1 = linear(z, count*256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256*count])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256*count],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256*count],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256*count],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Discriminator_SVHN(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix, z_logits,h5

def Discriminator_SVHN_WGAN(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits

def Discriminator_SVHN_WGAN_TeacherStudent(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        count = 2
        h1 = conv2d(image, count*64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, count*128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, count*256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, count*512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, count*1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits


def Discriminator_SVHN_conditional(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_class = linear(h5, 10, 'e_class')
        z_logits = linear(h5, 1, 'e_mix')
        return z_logits,z_class

def Discriminator_Celeba_WGAN(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits


def Encoder_SVHN(image, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.compat.v1.layers.batch_normalization(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = 100
        z_mean = linear(h5, 100, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Encoder_SVHN_LatentDim(image, name,dim, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        batch_size = 64
        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1))

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        continous_len = dim
        z_mean = linear(h5, continous_len, 'e_mean')
        z_log_sigma_sq = linear(h5, continous_len, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Generator_SubSVHN2_BatchEnsemble(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h3 = z_in

        h5 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8

def Generator_SharedSVHN2_BatchEnsemble(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        return h4


def Encoder_SVHN2(inputs,scopename, is_training=True,reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            #x = tf.reshape(inputs, [-1, 28, 28, 1])
            x = tf.reshape(inputs, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 4, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs,tf.nn.softmax(outputs)

def Generator_mnist(name,z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # ArchitecDiscriminator_Celebature : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope(name, reuse=reuse):

            # merge noise and code
            batch_size = 64
            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out

def Encoder_mnist(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 10

        is_training = True
        z_dim = 100
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,out_logit,softmaxValue

def Encoder_mnist2(image,name,dim, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 10

        is_training = True
        z_dim = 100
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_dim = dim
        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def classifier_mnist( x, is_training=True, reuse=False):
    batch_size = 128
    len_discrete_code = 10
    with tf.variable_scope("classifier", reuse=reuse):
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='c_fc2')
        out = tf.nn.softmax(out_logit)

    return out, out_logit

def classifier_mnist2( x,name, is_training=True, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        batch_size = 64
        len_discrete_code = 10
        with tf.variable_scope("classifier", reuse=reuse):
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
            net = tf.reshape(net, [batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

            net = lrelu(bn(linear(net, 64, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            out_logit = linear(net, len_discrete_code, scope='c_fc2')
            out = tf.nn.softmax(out_logit)

    return out, out_logit

def Discriminator_Mnist_ACGAN( x,name, batch_size = 64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        is_training = True
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
        classOutput = linear(net,10,scope='d_fc4')
        dis = linear(net, 1, scope='d_fc5')

    return dis,classOutput

def Discriminator_Mnist( x,name, batch_size = 64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        is_training = True
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
        out_logit = linear(net, 1, scope='d_fc4')
        out = tf.nn.sigmoid(out_logit)

    return out, out_logit, net

def Generator_SVHN_LGM(name,z_in, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        myScale = 1
        z = z_in
        kernel = 3
        # fully-connected layers
        h1 = linear(z, myScale * 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256 * myScale])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256 * myScale],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256 * myScale],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256 * myScale],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256 * myScale],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h8 = deconv2d(h5, [batch_size, 32, 32, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.sigmoid(h8)

        return h8


def Discriminator_Mnist2( x,z,name, batch_size = 64, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        is_training = True
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = tf.concat((net,z),axis=1)
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))

        len_discrete_code = 4
        out_logit2 = linear(net, len_discrete_code, scope='d_fc22')
        softmaxValue = tf.nn.softmax(out_logit2)

        out_logit = linear(net, 1, scope='d_fc4')
        out = tf.nn.sigmoid(out_logit)

    return out, out_logit, net,out_logit2,softmaxValue

def Generator_Celeba_Tanh(name, z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Generator_Celeba(z,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Generator_SubCeleba(z,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        '''
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        '''
        h1 = z

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

def Generator_Shared_Celeba(z,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        return h1



def Encoder_Celeba(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 4
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,out_logit,softmaxValue


def Generator_Celeba_Shared_BatchEnsemble(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        myT = 3
        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        kernel = 3
        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        kernel = 3
        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        return h6

def Generator_Celeba_Specific_BatchEnsemble(name,z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = 3
        h8 = deconv2d(z, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def Encoder_Celeba_Shared_CURL(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        size = 2
        z_dim = 256
        h1 = conv2d(image, 64*size, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128*size, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256*size, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        return h3

def Encoder_Celeba_Specific_CURL(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        size = 2
        h4 = conv2d(image, 512*size, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024*size, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq,h5


def Encoder_Celeba2(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 3
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def Encoder_Celeba2_Dim(image,name, batch_size=64,z_dim=256, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 3
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        z_mean = linear(h5, z_dim, 'e_mean')
        z_log_sigma_sq = linear(h5, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def Encoder_Celeba2_classifier(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        is_training = True
        len_discrete_code = 4
        net = lrelu(bn(linear(h5, 64, scope='e_fc11'), is_training=is_training, scope='c_bn1'))
        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue

def Discriminator_Celeba(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        return z_logits

def Discriminator_Celeba2(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = z_logits#tf.nn.sigmoid(z_logits)
        return z_mix,z_logits

def Discriminator_Celeba_Orginal(image,name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        z_dim = 256
        h1 = conv2d(image, 64, kernel, kernel, 2, 2, name='e_h1_conv')
        h1 = lrelu(h1)

        h2 = conv2d(h1, 128, kernel, kernel, 2, 2, name='e_h2_conv')
        h2 = lrelu(d_bn2(h2))

        h3 = conv2d(h2, 256, kernel, kernel, 2, 2, name='e_h3_conv')
        h3 = lrelu(d_bn3(h3))

        h4 = conv2d(h3, 512, kernel, kernel, 2, 2, name='e_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'e_h5_lin')
        h5 = lrelu(h5)

        z_logits = linear(h5, 1, 'e_mix')
        z_mix = tf.nn.sigmoid(z_logits)
        return z_mix,z_logits
