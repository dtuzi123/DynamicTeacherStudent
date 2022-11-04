import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
from glob import glob
from Basic_structure import *
from mnist_hand import *
from CIFAR10 import *
from skimage.measure import compare_ssim
import tf_slim as slim
import skimage as skimage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def file_name(file_dir):
    t1 = []
    file_dir = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)
    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def Image_classifier(inputs, scopename, is_training=True, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
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
    return outputs

def CodeImage_classifier(s, scopename, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        input = s

        # initializers
        w_init = tf.keras.initializers.glorot_normal()
        b_init = tf.constant_initializer(0.)
        n_hidden = 500
        keep_prob = 0.9

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 4
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def my_gumbel_softmax_sample(logits, cats_range, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    logits_with_noise = tf.nn.softmax(y / temperature)
    return logits_with_noise


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def My_Encoder_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq


def My_Classifier_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        # z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue


def MINI_Classifier(s, scopename, reuse=False):
    keep_prob = 1.0
    with tf.variable_scope(scopename, reuse=reuse):
        input = s
        n_output = 10
        n_hidden = 500
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 10
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y


# Create model of CNN with slim api

import FID_tf2 as fid2


class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 32
        self.input_width = 32
        self.c_dim = 3
        self.z_dim = 100
        self.len_discrete_code = 4
        self.epoch = 20

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        self.mnist_train_x = mnist_train_x
        self.mnist_train_y = np.zeros((np.shape(x_train)[0], 4))
        self.mnist_train_y[:, 0] = 1
        self.mnist_label = mnist_train_label
        self.mnist_label_test = mnist_label_test
        self.mnist_test_x = mnist_test
        self.mnist_test_y = np.zeros((np.shape(mnist_test)[0], 4))
        self.mnist_test_y[:, 0] = 1

        self.svhn_train_x = x_train
        self.svhn_train_y = np.zeros((np.shape(x_train)[0], 4))
        self.svhn_train_y[:, 1] = 1
        self.svhn_label = y_train
        self.svhn_label_test = y_test
        self.svhn_test_x = x_test
        self.svhn_test_y = np.zeros((np.shape(x_test)[0], 4))
        self.svhn_test_y[:, 1] = 1

        print(self.svhn_test_x[0:2])

        self.FashionTrain_x, self.FashionTrain_label, self.FashionTest_x, self.FashionTest_label = GiveFashion32()
        self.FashionTrain_y = np.zeros((np.shape(self.FashionTrain_x)[0], 4))
        self.FashionTrain_y[:, 2] = 1

        self.FashionTest_y = np.zeros((np.shape(self.FashionTest_x)[0], 4))
        self.FashionTest_y[:,2] = 1

        self.InverseFashionTrain_x, self.InverseFashionTrain_label, self.InverseFashionTest_x, self.InverseFashionTest_label = Give_InverseFashion32()
        self.InverseFashionTrain_y = np.zeros((np.shape(self.InverseFashionTrain_x)[0], 4))
        self.InverseFashionTrain_y[:, 3] = 1

        self.InverseFashionTest_y = np.zeros((np.shape(self.InverseFashionTest_x)[0], 4))
        self.InverseFashionTest_y[:, 3] = 1

        self.InverseMNISTTrain_x, self.InverseMNISTTrain_label, self.InverseMNISTTest_x, self.InverseMNISTTest_label = GiveMNIST32()

        self.CurrentExpertIndex = 0
        self.GeneratorArr = []

    def Create_subloss(self, G, name):
        name = "discriminator1"
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G
        d_hat = Discriminator_SVHN_WGAN(x_hat, name, reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        return ddx

    def Build_NewExpert(self):
        newIndex = np.shape(self.GeneratorArr)[0]
        newIndex = newIndex + 1
        str1 = "GAN_generator" + str(newIndex)
        discStr = "discriminator1"  # +str(newIndex)

        gan_code = self.z
        G1 = Generator_SVHN(str1, gan_code, reuse=False)
        self.GeneratorArr.append(G1)

        D_real_logits = Discriminator_SVHN_WGAN(self.inputs, discStr, reuse=True)

        # output of D for fake images
        D_fake_logits1 = Discriminator_SVHN_WGAN(G1, discStr, reuse=True)

        self.g_loss1 = -tf.reduce_mean(D_fake_logits1)
        self.d_loss1 = -tf.reduce_mean(D_real_logits) + tf.reduce_mean(D_fake_logits1)
        self.d_loss1 = self.d_loss1 + self.Create_subloss(G1, discStr)

        T_vars = tf.trainable_variables()
        generator_var = [var for var in T_vars if var.name.startswith(str1)]
        disc_var = [var for var in T_vars if var.name.startswith(discStr)]

        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.d_optim1 = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
                .minimize(self.d_loss1, var_list=disc_var)
            self.g_optim1 = tf.train.RMSPropOptimizer(learning_rate=1e-4) \
                .minimize(self.g_loss1, var_list=generator_var)

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        self.sess.run(tf.variables_initializer(not_initialized_vars))

    def Select_Expert(self, index):
        str1 = "GAN_generator" + str(index)
        discStr = "discriminator1"# + str(index)

        gan_code = self.z
        G1 = Generator_SVHN(str1, gan_code, reuse=True)

        D_real_logits = Discriminator_SVHN_WGAN(self.inputs, discStr, reuse=True)

        # output of D for fake images
        D_fake_logits1 = Discriminator_SVHN_WGAN(G1, discStr, reuse=True)

        self.g_loss1 = -tf.reduce_mean(D_fake_logits1)
        self.d_loss1 = -tf.reduce_mean(D_real_logits) + tf.reduce_mean(D_fake_logits1)
        self.d_loss1 = self.d_loss1 + self.Create_subloss(G1, discStr)

        T_vars = tf.trainable_variables()
        generator_var = [var for var in T_vars if var.name.startswith(str1)]
        disc_var = [var for var in T_vars if var.name.startswith(discStr)]

        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            self.d_optim1 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss1, var_list=disc_var)
            self.g_optim1 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss1, var_list=generator_var)

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        self.sess.run(tf.variables_initializer(not_initialized_vars))

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])
        self.weights = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.index = tf.placeholder(tf.int32, [self.batch_size])
        self.gan_inputs = tf.placeholder(tf.float32, [bs] + image_dims)
        self.gan_domain = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.gan_domain_labels = tf.placeholder(tf.float32, [self.batch_size, 1])

        domain_labels = tf.argmax(self.gan_domain, 1)

        # GAN networks
        gan_code = self.z
        G1 = Generator_SVHN("GAN_generator1", gan_code, reuse=False)

        self.GeneratorArr.append(G1)
        ## 1. GAN Loss
        # output of D for real images
        D_real_logits = Discriminator_SVHN_WGAN(self.inputs, "discriminator1", reuse=False)

        # output of D for fake images
        D_fake_logits1 = Discriminator_SVHN_WGAN(G1, "discriminator1", reuse=True)

        self.g_loss1 = -tf.reduce_mean(D_fake_logits1)

        self.d_loss1 = -tf.reduce_mean(D_real_logits) + tf.reduce_mean(D_fake_logits1)

        self.d_loss1 = self.d_loss1 + self.Create_subloss(G1, "discriminator1")

        self.GAN_gen1 = G1

        G_all = self.gan_inputs

        z_dim = 150
        # encoder continoual information
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(G_all, "encoder", batch_size=64, reuse=False)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain")
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        self.domain_classloss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=domain_logit, labels=self.gan_domain))

        y_labels = tf.argmax(self.gan_domain, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=False)

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1 - y_labels) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        KL_divergence1_normal = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1_normal = tf.reduce_mean(KL_divergence1_normal)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - G_all), [1, 2, 3]))

        vaeloss1 = reconstruction_loss1 + KL_divergence1
        self.vaeLoss = vaeloss1
        self.vaeloss1_normal = reconstruction_loss1 + KL_divergence1_normal

        self.studentReco = VAE1
        self.stuRecoLoss = reconstruction_loss1
        # Get VAE loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith('GAN_generator1')]
        VAE_encoder_vars = [var for var in T_vars if var.name.startswith('encoder')]
        VAE_encoder_domain_vars = [var for var in T_vars if var.name.startswith('encoder_domain')]
        VAE_generator_vars = [var for var in T_vars if var.name.startswith('VAE_Generator')]

        vae_vars = VAE_encoder_vars + VAE_generator_vars + VAE_encoder_domain_vars
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            '''
            self.d_optim1 = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9) \
                .minimize(self.d_loss1, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9) \
                .minimize(self.g_loss1, var_list=GAN_generator_vars1)
            '''
            self.d_optim1 = tf.train.RMSPropOptimizer(learning_rate=0.0001) \
                .minimize(self.d_loss1, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.RMSPropOptimizer(learning_rate=0.0001) \
                .minimize(self.g_loss1, var_list=GAN_generator_vars1)
            self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=vae_vars)
            self.domain_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.domain_classloss, var_list=VAE_encoder_domain_vars)

        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "label_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def DomainPredict(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1,
                                                                            dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        label_softmax = tf.nn.softmax(discrete_real)
        predictions = tf.argmax(label_softmax, 1, name="domainPredictions")
        return predictions

    def Give_predictedLabels(self, testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions)
        return totalPredictions

    def Give_DomainpredictedLabels(self, testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.DomainPredict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions, 4)
        return totalPredictions

    def Calculate_DomainAcc(self, testX, testY):
        # testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.DomainPredict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def Calculate_accuracy(self, testX, testY):
        # testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def Give_RealReconstruction(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        return reconstruction_loss1

    def Give_Elbo(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        y_labels = tf.argmax(discrete_real, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1 - y_labels) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        return reconstruction_loss1 + KL_divergence1

    def Calculate_ReconstructionErrors(self, testX):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Give_RealReconstruction()
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError

    def Calculate_Elbo(self, testX):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        myPro = self.Give_Elbo()
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError

    def test(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        z_dim = 150
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", reuse=True)

        label_softmax = tf.nn.softmax(domain_logit)
        predictions = tf.argmax(label_softmax, 1, name="predictions")

        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        testX = np.concatenate((self.mnist_test_x, self.svhn_test_x, self.FashionTest_x, self.InverseFashionTest_x),
                               axis=0)
        testY = np.concatenate((self.mnist_test_y, self.svhn_test_y, self.FashionTest_y, self.InverseFashionTest_y),
                               axis=0)
        testY = np.concatenate((self.mnist_test_y, self.svhn_test_y, self.FashionTest_y, self.InverseFashionTest_y),
                               axis=0)

        with tf.Session(config=config) as sess:
            self.saver = tf.train.Saver()

            self.Parameter_GetCount()

            '''
            self.InverseMNISTTrain_x = tf.image.flip_left_right(self.InverseMNISTTrain_x)
            self.InverseMNISTTest_x = tf.image.flip_left_right(self.InverseMNISTTest_x)
            self.InverseMNISTTrain_x = self.InverseMNISTTrain_x.eval()
            self.InverseMNISTTest_x = self.InverseMNISTTest_x.eval()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/LifelongMixtureGANs_Unsupervised_32')

            batch = np.concatenate((self.mnist_test_x[0:self.batch_size], self.svhn_test_x[0:self.batch_size],
                                    self.FashionTest_x[0:self.batch_size],
                                    self.InverseFashionTest_x[0:self.batch_size],self.InverseMNISTTest_x[0:self.batch_size]), axis=0)

            batch = self.mnist_test_x
            #batch = np.concatenate((self.mnist_test_x[0:self.batch_size], self.svhn_test_x[0:self.batch_size],
            #                        self.FashionTest_x[0:self.batch_size]), axis=0)

            index = [i for i in range(np.shape(batch)[0])]
            random.shuffle(index)
            batch = batch[index]
            batch = batch[0:self.batch_size]

            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            gan1, gan2, gan3, gan4 = self.sess.run(
                [self.GAN_gen1, self.GAN_gen2, self.GAN_gen3, self.GAN_gen4],
                feed_dict={self.inputs: batch, self.z: batch_z, self.gan_inputs: batch})

            myReco = self.sess.run(VAE1, feed_dict={self.inputs: batch})

            ims("results/" + "TestReal_five_small" + str(0) + ".png", merge2(batch[:5], [1, 5]))
            ims("results/" + "TestReco_five_small" + str(0) + ".png", merge2(myReco[:5], [1, 5]))
            ims("results/" + "TestGAN_five_small" + str(0) + ".png", merge2(gan1[:5], [1, 5]))
            ims("results/" + "TestGAN_five_small" + str(1) + ".png", merge2(gan2[:5], [1, 5]))
            ims("results/" + "TestGAN_five_small" + str(2) + ".png", merge2(gan3[:5], [1, 5]))
            ims("results/" + "TestGAN_five_small" + str(3) + ".png", merge2(gan4[:5], [1, 5]))
            '''

            '''
            Make domain inference
            '''

            '''
            mnistDomain = np.zeros((10000,4))
            mnistDomain[:,0] = 1

            SvhnDomain = np.zeros((np.shape(self.svhn_test_x)[0], 4))
            SvhnDomain[:, 1] = 1

            FashionDomain = np.zeros((10000, 4))
            FashionDomain[:, 2] = 1

            IFashionDomain = np.zeros((10000, 4))
            IFashionDomain[:, 2] = 1

            RMNISTDomain = np.zeros((10000, 4))
            RMNISTDomain[:, 1] = 1

            acc1 = self.Calculate_DomainAcc(self.mnist_test_x,mnistDomain)
            print(acc1)
            acc1 = self.Calculate_DomainAcc(self.svhn_test_x, SvhnDomain)
            print(acc1)
            acc1 = self.Calculate_DomainAcc(self.FashionTest_x, FashionDomain)
            print(acc1)
            acc1 = self.Calculate_DomainAcc(self.InverseFashionTest_x, IFashionDomain)
            print(acc1)
            acc1 = self.Calculate_DomainAcc(self.InverseMNISTTest_x, RMNISTDomain)
            print(acc1)

            print('log-likelihood')

            mnistError = self.Calculate_Elbo(self.mnist_test_x)
            fashionError = self.Calculate_Elbo(self.FashionTest_x)
            svhnError = self.Calculate_Elbo(self.svhn_test_x)
            IFashionError = self.Calculate_Elbo(self.InverseFashionTest_x)
            IMNISTError = self.Calculate_Elbo(self.InverseMNISTTest_x)

            sum1 = mnistError+fashionError+svhnError+IFashionError+IMNISTError
            sum1 = sum1 / 5.0

            print(mnistError)
            print('\n')
            print(svhnError)
            print('\n')
            print(fashionError)
            print('\n')
            print(IFashionError)
            print('\n')
            print(IMNISTError)
            print('\n')
            print(sum1)
            '''

    def SelectExpert(self, testX, weights, domainState, taskIndex):
        gan_code = self.z

        batch_labels = np.random.multinomial(1,
                                             self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
                                             size=[self.batch_size])
        # update GAN
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        G1 = Generator_SVHN("GAN_generator1", gan_code, reuse=True)
        G2 = Generator_SVHN("GAN_generator2", gan_code, reuse=True)
        G3 = Generator_SVHN("GAN_generator3", gan_code, reuse=True)
        G4 = Generator_SVHN("GAN_generator4", gan_code, reuse=True)

        D_real_logits = Discriminator_SVHN_WGAN(self.inputs, "discriminator", reuse=True)
        # output of D for fake images
        D_fake_logits1 = Discriminator_SVHN_WGAN(G1, "discriminator", reuse=True)
        D_fake_logits2 = Discriminator_SVHN_WGAN(G2, "discriminator", reuse=True)
        D_fake_logits3 = Discriminator_SVHN_WGAN(G3, "discriminator", reuse=True)
        D_fake_logits4 = Discriminator_SVHN_WGAN(G4, "discriminator", reuse=True)

        d_loss1 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits1))
        d_loss2 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits2))
        d_loss3 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits3))
        d_loss4 = tf.abs(tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits4))

        d_loss1_, d_loss2_, d_loss3_, d_loss4_ = self.sess.run([d_loss1, d_loss2, d_loss3, d_loss4],
                                                               feed_dict={self.inputs: testX,
                                                                          self.z: batch_z})
        d_loss1_ = d_loss1_ + (1 - weights[0, 0]) * 10000
        d_loss2_ = d_loss2_ + (1 - weights[0, 1]) * 10000
        d_loss3_ = d_loss3_ + (1 - weights[0, 2]) * 10000
        d_loss4_ = d_loss4_ + (1 - weights[0, 3]) * 10000

        score = []

        score.append(d_loss1_)
        score.append(d_loss2_)
        score.append(d_loss3_)
        score.append(d_loss4_)
        index = score.index(min(score))
        if index == 0:
            weights[:, 0] = 0
            tmp = domainState[0]
            domainState[0] = taskIndex
            # domainState[taskIndex] = tmp
        elif index == 1:
            weights[:, 1] = 0
            # tmp = domainState[1]
            domainState[1] = taskIndex
            # domainState[taskIndex] = tmp
        elif index == 2:
            weights[:, 2] = 0
            # tmp = domainState[2]
            domainState[2] = taskIndex
            # domainState[taskIndex] = tmp
        elif index == 3:
            weights[:, 3] = 0
            # tmp = domainState[3]
            domainState[3] = taskIndex
            # domainState[taskIndex] = tmp

        return weights, domainState

    def SelectGANs_byIndex(self, index):
        if index == 1:
            return self.GAN_gen1
        elif index == 2:
            return self.GAN_gen2
        elif index == 3:
            return self.GAN_gen3
        elif index == 4:
            return self.GAN_gen4

    def Parameter_GetCount(self):
        total_parameters = 0
        totalTraining = tf.trainable_variables()

        T_vars = tf.trainable_variables()

        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator')]
        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith('GAN_generator1')]
        GAN_generator_vars2 = [var for var in T_vars if var.name.startswith('GAN_generator2')]
        GAN_generator_vars3 = [var for var in T_vars if var.name.startswith('GAN_generator3')]
        VAE_encoder_vars = [var for var in T_vars if var.name.startswith('encoder')]
        VAE_encoder_domain_vars = [var for var in T_vars if var.name.startswith('encoder_domain')]
        VAE_generator_vars = [var for var in T_vars if var.name.startswith('VAE_Generator')]

        myTotal = discriminator_vars1 + GAN_generator_vars1 + GAN_generator_vars2 + GAN_generator_vars3 + VAE_encoder_vars + VAE_encoder_domain_vars + VAE_generator_vars
        VAE_parameters = VAE_encoder_vars + VAE_encoder_domain_vars + VAE_generator_vars

        for variable in myTotal:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print("Total parameters:")
        print(total_parameters)

        total_parameters = 0
        for variable in VAE_parameters:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print("VAE parameters:")
        print(total_parameters)

    def GenerateSamplesBySelect(self, n, index):
        myGANs = self.GeneratorArr[index - 1]
        a = int(n / self.batch_size)
        myArr = []
        for i in range(a):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            aa = self.sess.run(myGANs, feed_dict={self.z: batch_z})
            for t in range(self.batch_size):
                myArr.append(aa[t])
        myArr = np.array(myArr)
        return myArr

    def Evaluation(self, testData):

        G_all = self.gan_inputs
        z_dim = 150
        # encoder continoual information
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(G_all, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain", True)
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        y_labels = tf.argmax(discrete_real, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_SVHN("VAE_Generator", code, reuse=True)

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1 - y_labels) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - G_all), [1, 2, 3]))

        vaeloss1 = reconstruction_loss1 + KL_divergence1

        count = np.shape(testData)[0]
        count = int(count / self.batch_size)
        ssimSum = 0
        psnrSum = 0
        recoLossSum = 0
        elboSum = 0
        for i in range(count):
            batch = testData[i * self.batch_size:(i + 1) * self.batch_size]
            reco = self.sess.run(VAE1, feed_dict={self.gan_inputs: batch})
            mySum2 = 0
            # Calculate SSIM
            for t in range(self.batch_size):
                # ssim_none = ssim(g[t], r[t], data_range=np.max(g[t]) - np.min(g[t]))
                ssim_none = compare_ssim(batch[t], reco[t], multichannel=True)
                mySum2 = mySum2 + ssim_none
            mySum2 = mySum2 / self.batch_size
            ssimSum = ssimSum + mySum2

            # Calculate PSNR
            mySum2 = 0
            for t in range(self.batch_size):
                measures = skimage.measure.compare_psnr(batch[t], reco[t], data_range=np.max(reco[t]) - np.min(reco[t]))
                mySum2 = mySum2 + measures
            mySum2 = mySum2 / self.batch_size
            psnrSum = psnrSum + mySum2

            recoLoss = self.sess.run(reconstruction_loss1, feed_dict={self.gan_inputs: batch})
            recoLossSum = recoLossSum + recoLoss

            ELBO = self.sess.run(vaeloss1, feed_dict={self.gan_inputs: batch})
            elboSum = elboSum + ELBO

        ssimSum = ssimSum / count
        psnrSum = psnrSum / count
        recoLossSum = recoLossSum / count
        elboSum = elboSum / count

        return ssimSum, psnrSum, recoLossSum, elboSum

    def Calculate_FID_Score(self, nextTaskIndex):
        fid2.session = self.sess

        if nextTaskIndex == 0:
            nextTrainX = self.mnist_train_x
        elif nextTaskIndex == 1:
            nextTrainX = self.svhn_train_x
        elif nextTaskIndex == 2:
            nextTrainX = self.FashionTrain_x
        elif nextTaskIndex == 3:
            nextTrainX = self.InverseFashionTrain_x
        elif nextTaskIndex == 4:
            nextTrainX = self.InverseMNISTTrain_x

        myCount = 1000
        realImages = nextTrainX[0:myCount]
        realImages = np.transpose(realImages, (0, 3, 1, 2))
        realImages = ((realImages + 1.0) * 255) / 2.0

        # Calculate FID
        fidArr = []
        for tIndex in range(np.shape(self.GeneratorArr)[0]):
            fakeImages = []
            myGANs = self.GeneratorArr[tIndex]
            tt = int(myCount / self.batch_size)
            for i in range(tt):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                aa = self.sess.run(myGANs, feed_dict={self.z: batch_z})
                if i == 0:
                    bachFake = aa
                for j in range(self.batch_size):
                    fakeImages.append(aa[j])

            fakeImages = np.array(fakeImages)
            realImages = realImages[0:np.shape(fakeImages)[0]]
            fakeImages = np.transpose(fakeImages, (0, 3, 1, 2))
            fakeImages = ((fakeImages + 1.0) * 255) / 2.0

            fidScore = fid2.get_fid(realImages, fakeImages)
            print(fidScore)
            fidArr.append(fidScore)

        # Compare FID
        minIndex = fidArr.index(min(fidArr))
        minFid = min(fidArr)
        minIndex = minIndex + 1

        return minIndex, minFid

    def train(self):

        taskCount = 4

        config = tf.ConfigProto(allow_soft_placement=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config.gpu_options.allow_growth = True

        self.componentCount = 1
        self.currentComponent = 1
        self.fid_hold = 250
        self.IsAdd = 0

        isFirstStage = True
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion_invariant')

            # saver to save model
            self.saver = tf.train.Saver()
            ExpertWeights = np.ones((self.batch_size, 4))

            DomainState = np.zeros(4).astype(np.int32)
            DomainState[0] = 0
            DomainState[1] = 1
            DomainState[2] = 2
            DomainState[3] = 3

            self.InverseMNISTTrain_x = tf.image.flip_left_right(self.InverseMNISTTrain_x)
            self.InverseMNISTTest_x = tf.image.flip_left_right(self.InverseMNISTTest_x)
            self.InverseMNISTTrain_x = self.InverseMNISTTrain_x.eval()
            self.InverseMNISTTest_x = self.InverseMNISTTest_x.eval()

            taskCount = 5
            for taskIndex in range(taskCount):
                if taskIndex == 0:
                    currentTrainX = self.mnist_train_x
                elif taskIndex == 1:
                    currentTrainX = self.svhn_train_x
                elif taskIndex == 2:
                    currentTrainX = self.FashionTrain_x
                elif taskIndex == 3:
                    currentTrainX = self.InverseFashionTrain_x
                elif taskIndex == 4:
                    currentTrainX = self.InverseMNISTTrain_x

                currentY = np.zeros((np.shape(currentTrainX)[0], 4))
                currentY[:, self.currentComponent - 1] = 1

                if self.IsAdd == 0:
                    if taskIndex != 0:
                        oldX = self.GenerateSamplesBySelect(50000, self.currentComponent)
                        currentTrainX = np.concatenate((currentTrainX, oldX), axis=0)
                        currentY = np.zeros((np.shape(currentTrainX)[0], 4))
                        currentY[:, self.currentComponent - 1] = 1
                    '''
                    currentTrainX = self.cifar_train_x
                    currentTrainY = self.CifarTrain_y
                    currentTrain_labels = self.cifar_train_label
                    '''

                dataX = currentTrainX
                dataY = currentY
                n_examples = np.shape(dataX)[0]

                start_epoch = 0
                start_batch_id = 0
                self.num_batches = int(n_examples / self.batch_size)

                mnistAccuracy_list = []
                mnistFashionAccuracy_list = []

                # loop for epoch
                start_time = time.time()
                for epoch in range(start_epoch, self.epoch):
                    count = 0
                    # Random shuffling
                    index = [i for i in range(n_examples)]
                    random.shuffle(index)
                    dataX = dataX[index]
                    dataY = dataY[index]
                    counter = 0

                    # get batch data
                    for idx in range(start_batch_id, self.num_batches):
                        batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_y = dataY[idx * self.batch_size:(idx + 1) * self.batch_size]

                        # update GAN
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                        dataIndex = [i for i in range(self.batch_size)]
                        random.shuffle(dataIndex)

                        _, d_loss = self.sess.run([self.d_optim1, self.d_loss1],
                                                  feed_dict={self.inputs: batch_images,
                                                             self.z: batch_z, self.y: batch_y
                                                      , self.weights: ExpertWeights,
                                                             self.index: dataIndex, self.gan_inputs: batch_images})

                        if idx % 5 == 0:
                            # update G and Q network
                            _, g_loss = self.sess.run(
                                [self.g_optim1, self.g_loss1],
                                feed_dict={self.inputs: batch_images, self.z: batch_z, self.y: batch_y,
                                           self.weights: ExpertWeights, self.index: dataIndex,
                                           self.gan_inputs: batch_images})

                        if taskIndex > 0:
                            ganSamples = batch_images
                            gan_domain = np.zeros((self.batch_size, 4))
                            gan_domain[:, self.currentComponent-1] = 1

                            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                            for i in range(np.shape(self.GeneratorArr)[0]):
                                newGan_domain = np.zeros((self.batch_size, 4))
                                newGan_domain[:, i] = 1
                                gan_domain = np.concatenate((gan_domain, newGan_domain), axis=0)

                                gan1 = self.sess.run(self.GeneratorArr[i],feed_dict={self.z:batch_z})
                                ganSamples = np.concatenate((ganSamples, gan1), axis=0)

                            gan_domain_labels = [np.argmax(one_hot) for one_hot in gan_domain]

                        else:
                            ganSamples = batch_images
                            gan_domain = np.zeros((np.shape(ganSamples)[0], 4))
                            gan_domain[:, 0] = 1
                            gan_domain_labels = [np.argmax(one_hot) for one_hot in gan_domain]

                        gan_domain_labels = np.array(gan_domain_labels)
                        gan_domain_labels = np.reshape(gan_domain_labels, (-1, 1))
                        index = [i for i in range(np.shape(ganSamples)[0])]
                        random.shuffle(index)
                        ganSamples = ganSamples[index]
                        ganSamples = ganSamples[0:self.batch_size]
                        gan_domain = gan_domain[index]
                        gan_domain = gan_domain[0:self.batch_size]
                        gan_domain_labels = gan_domain_labels[index]
                        gan_domain_labels = gan_domain_labels[0:self.batch_size]

                        # update G and Q network
                        _, vaeLoss, _ = self.sess.run(
                            [self.vae_optim, self.vaeLoss, self.domain_optim],
                            feed_dict={self.inputs: batch_images, self.z: batch_z, self.y: batch_y,
                                       self.weights: ExpertWeights, self.index: dataIndex,
                                       self.gan_inputs: ganSamples, self.gan_domain: gan_domain,
                                       self.gan_domain_labels: gan_domain_labels})

                        # display training status
                        counter += 1
                        print(
                            "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                            % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, 0, 0))

                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    gan = self.sess.run(self.GeneratorArr[np.shape(self.GeneratorArr)[0] - 1],
                                        feed_dict={self.z: batch_z})
                    ims("results/" + "Test" + str(epoch) + ".png", merge2(gan[:64], [8, 8]))

                nextTaskIndex = taskIndex + 1
                if nextTaskIndex < 5:
                    minIndex,minFID = self.Calculate_FID_Score(nextTaskIndex)

                    self.fid_hold = 50
                    if minFID > self.fid_hold:#add a new GANs
                        self.Build_NewExpert()
                        self.IsAdd = 1
                    else:
                        #continous to use the current GANs
                        self.IsAdd = 0
                        self.Select_Expert(minIndex)
                        self.currentComponent = minIndex

                    print(np.shape(self.GeneratorArr)[0])

            for k in range(np.shape(self.GeneratorArr)[0]):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                gan = self.sess.run(self.GeneratorArr[k], feed_dict={self.z: batch_z})
                ims("results/" + "LTS_GAN" + str(k) + ".png", merge2(gan[:64], [8, 8]))

            ssim1,psnr1,reco1,elbo1 = self.Evaluation(self.mnist_test_x)
            ssim2,psnr2,reco2,elbo2 = self.Evaluation(self.svhn_test_x)
            ssim3,psnr3,reco3,elbo3 = self.Evaluation(self.FashionTest_x)
            ssim4,psnr4,reco4,elbo4 = self.Evaluation(self.InverseFashionTest_x)
            ssim5,psnr5,reco5,elbo5 = self.Evaluation(self.InverseMNISTTest_x)

            ssimSum = ssim1 + ssim2 +ssim3 +ssim4+ssim5
            psnrSum = psnr1+psnr2+psnr3+psnr4+psnr5
            recoSum = reco1 +reco2+reco3+reco4+reco5
            elboSum = elbo1+elbo2+elbo3+elbo4+elbo5

            ssimSum = ssimSum/5.0
            psnrSum = psnrSum /5.0
            recoSum = recoSum/5.0
            elboSum = elboSum/5.0

            print(
                "SSIM:%.8f. PSNR:%.8f, Reco:%.8f,ELBO:%.8f" \
                % (ssim1, psnr1, reco1, elbo1))

            print(
                "SSIM:%.8f. PSNR:%.8f, Reco:%.8f,ELBO:%.8f" \
                % (ssim2, psnr2, reco2, elbo2))

            print(
                "SSIM:%.8f. PSNR:%.8f, Reco:%.8f,ELBO:%.8f" \
                % (ssim3, psnr3, reco3, elbo3))

            print(
                "SSIM:%.8f. PSNR:%.8f, Reco:%.8f,ELBO:%.8f" \
                % (ssim4, psnr4, reco4, elbo4))

            print(
                "SSIM:%.8f. PSNR:%.8f, Reco:%.8f,ELBO:%.8f" \
                % (ssim5, psnr5, reco5, elbo5))

            print(
                "SSIM:%.8f. PSNR:%.8f, Reco:%.8f,ELBO:%.8f" \
                % (ssimSum, psnrSum, recoSum, elboSum))

infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
# infoMultiGAN.test()
