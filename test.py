import chainer
import numpy as np
import tensorflow as tf

import chainer.functions as F
import chainer.links as L

from chainercv.links.connection.conv_2d_activ import Conv2DActiv
from chainercv.links.model.pickable_sequential_chain import \
    PickableSequentialChain

import mobilenet_v1_tf

### TENSORFLOW ###
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model_tf, _ = mobilenet_v1_tf.mobilenet_v1(x, num_classes=1001, is_training=False)

saver = tf.train.Saver()

### CHAINER ###

class ConvBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=True, initialW=None, initial_bias=None,
                 activation=F.relu, **kwargs):
        self.activation = activation

        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize,
                                             stride, pad, nobias, initialW,
                                             initial_bias)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):

        if self.activation is not None:
            h = self.activation(self.bn(self.conv(x)))

        else:
            h = self.bn(self.conv(x))

        return h

class DWSeparableConvBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=True, initialW=None, initial_bias=None,
                 activation=F.relu, **kwargs):
        self.activation = activation

        super(DWSeparableConvBlock, self).__init__()
        with self.init_scope():
            self.depthwise = L.DepthwiseConvolution2D(in_channels, 1, ksize,
                                                      stride, pad, nobias,
                                                      initialW, initial_bias)
            self.depthwise_bn = L.BatchNormalization(in_channels)
            self.pointwise = L.Convolution2D(in_channels, out_channels, 1,
                                             1, 0, nobias, initialW,
                                             initial_bias)
            self.pointwise_bn = L.BatchNormalization(out_channels)

    def __call__(self, x):

        h = self.activation(self.depthwise_bn(self.depthwise(x)))

        if self.activation is not None:
            h = self.activation(self.pointwise_bn(self.pointwise(h)))

        else:
            h = self.pointwise_bn(self.pointwise(h))

        return h

class LogitsBlock(chainer.Chain):
    def __init__(self, in_channels, num_classes, dropout_keep_prob):
        self.dropout_keep_prob = dropout_keep_prob
        super(LogitsBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, num_classes, 1, 1, 0)

    def __call__(self, features):
        h = F.average_pooling_2d(features, features.shape[2:4], 1, 0)
        h = F.dropout(h, self.dropout_keep_prob)
        h = self.conv(h)
        h = F.reshape(h, (h.shape[0], h.shape[1]))

        return h

class MobileNet_V1(PickableSequentialChain):
    def __init__(self, num_classes=1001, dropout_keep_prob=0.999, min_depth=8,
                 depth_multiplier=1.0, prediction_fn=F.softmax,
                 global_pool=False):
        ch = np.array([32, 64, 128, 256, 512, 1024])
        ch = ch * depth_multiplier
        ch = np.maximum(ch, depth_multiplier)
        ch = ch.astype(np.int32)

        super(MobileNet_V1, self).__init__()
        with self.init_scope():
            self.conv_0 = ConvBlock(3, ch[0], ksize=3, stride=2, pad=1,
                                    nobias=True)
            self.conv_1 = DWSeparableConvBlock(ch[0], ch[1], ksize=3, pad=1)
            self.conv_2 = DWSeparableConvBlock(ch[1], ch[1], ksize=3, pad=1,
                                               stride=2)
            self.conv_3 = DWSeparableConvBlock(ch[1], ch[2], ksize=3, pad=1)
            self.conv_4 = DWSeparableConvBlock(ch[2], ch[2], ksize=3, pad=1,
                                               stride=2)
            self.conv_5 = DWSeparableConvBlock(ch[2], ch[3], ksize=3, pad=1)
            self.conv_6 = DWSeparableConvBlock(ch[3], ch[3], ksize=3, pad=1,
                                               stride=2)
            self.conv_7 = DWSeparableConvBlock(ch[3], ch[4], ksize=3, pad=1)
            self.conv_8 = DWSeparableConvBlock(ch[4], ch[4], ksize=3, pad=1)
            self.conv_9 = DWSeparableConvBlock(ch[4], ch[4], ksize=3, pad=1)
            self.conv_10 = DWSeparableConvBlock(ch[4], ch[4], ksize=3, pad=1)
            self.conv_11 = DWSeparableConvBlock(ch[4], ch[4], ksize=3, pad=1)
            self.conv_12 = DWSeparableConvBlock(ch[4], ch[4], ksize=3, pad=1,
                                               stride=2)
            self.conv_13 = DWSeparableConvBlock(ch[4], ch[5], ksize=3, pad=1)
            self.fc1 = LogitsBlock(ch[5], num_classes, dropout_keep_prob)
            self.prob = prediction_fn

model_ch = MobileNet_V1()

with tf.Session() as sess:
    saver.restore(sess, 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt')
    variables = tf.global_variables()

    for i in variables:
        print(i.name, i.shape)

    writer = tf.summary.FileWriter('logs', sess.graph)
    writer.close()

