import chainer
import numpy as np

import chainer.functions as F
import chainer.links as L

from chainercv.links.model.pickable_sequential_chain import \
    PickableSequentialChain

def _get_same_padding(in_shape, ksize, stride):
    ''' Retrieves the inputs necessary for chainer.functions.pad to return
    padding similar to that of Tensorflow. By default, if the amount of padding
    required is an odd number, Tensorflow will pad the end of each index with
    the odd value. This is in contrast to other implementations such as cudNN
    and Caffe, which pad both sides equally by the odd amount
    '''

    in_h = in_shape[0]
    in_w = in_shape[1]
    if in_shape[0] % stride == 0:
        pad_h = max(ksize - stride, 0)
    else:
        pad_h = max(ksize - (in_shape[0] % stride), 0)
    if in_shape[0] % stride == 0:
        pad_w = max(ksize - stride, 0)
    else:
        pad_w = max(ksize - (in_shape[1] % stride), 0)

    pad_top = pad_h // 2
    pad_lft = pad_w // 2
    pad_bot = pad_h - pad_top
    pad_rgt = pad_w - pad_lft

    return ((pad_top, pad_bot), (pad_lft, pad_rgt))

class TFConvBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=True, initialW=None, initial_bias=None,
                 activation_fn=F.relu, **kwargs):
        self.activation_fn = activation_fn
        self.ksize = ksize
        self.stride = stride

        super(TFConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize,
                                          stride, 0, nobias, initialW,
                                          initial_bias)
            self.bn = L.BatchNormalization(out_channels, decay=0.9997,
                                                  eps=0.001)

    def __call__(self, x):
        pd = _get_same_padding(x.shape[2:4], self.ksize, self.stride)
        pd = ((0, 0), (0, 0), pd[0], pd[1])
        h = F.pad(x, pd, mode='constant', constant_values=0)
        h = self.conv(h)
        h = self.bn(h)
        if self.activation_fn is not None:
            h = self.activation_fn(h)

        return h

class TFDWSeparableConvBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=True, initialW=None, initial_bias=None,
                 activation_fn=F.relu, **kwargs):
        self.activation_fn = activation_fn
        self.ksize = ksize
        self.stride = stride

        super(TFDWSeparableConvBlock, self).__init__()
        with self.init_scope():
            self.dw = L.DepthwiseConvolution2D(in_channels, 1, ksize, stride, 0,
                                               nobias, initialW, initial_bias)
            self.dw_bn = L.BatchNormalization(in_channels, decay=0.9997,
                                                            eps=0.001)
            self.pw = L.Convolution2D(in_channels, out_channels, 1, 1, 0, 
                                      nobias, initialW, initial_bias)
            self.pw_bn = L.BatchNormalization(out_channels, decay=0.9997,
                                                            eps=0.001)

    def __call__(self, x):
        pd = _get_same_padding(x.shape[2:4], self.ksize, self.stride)
        pd = ((0, 0), (0, 0), pd[0], pd[1])
        h = F.pad(x, pd, mode='constant', constant_values=0)

        h = self.dw(h)
        h = self.dw_bn(h)
        h = self.activation_fn(h)

        h = self.pw(h)
        h = self.pw_bn(h)
        h = self.activation_fn(h)

        return h

class TFLogitsBlock(chainer.Chain):
    def __init__(self, in_channels, num_classes, dropout_keep_prob):
        self.dropout_keep_prob = dropout_keep_prob

        super(TFLogitsBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, num_classes, 1, 
                                        nobias=False)

    def __call__(self, x):
        h = F.average_pooling_2d(x, x.shape[2:4], 1, 0)
        h = F.dropout(h, self.dropout_keep_prob)
        h = self.conv(h)

        return h

class MobilenetV1(PickableSequentialChain):
    def __init__(self, num_classes=1001, dropout_keep_prob=0.999, min_depth=8,
                 depth_multiplier=1.0, prediction_fn=F.softmax,
                 global_pool=False):
        ch = np.array([32, 64, 128, 256, 512, 1024])
        ch = ch * depth_multiplier
        ch = np.maximum(ch, min_depth)
        ch = ch.astype(np.int32)

        super(MobilenetV1, self).__init__()
        with self.init_scope():
            self.conv2d_0 = TFConvBlock(3, ch[0], 3, stride=2)
            self.conv2d_1 = TFDWSeparableConvBlock(ch[0], ch[1], 3)
            self.conv2d_2 = TFDWSeparableConvBlock(ch[1], ch[2], 3, stride=2)
            self.conv2d_3 = TFDWSeparableConvBlock(ch[2], ch[2], 3)
            self.conv2d_4 = TFDWSeparableConvBlock(ch[2], ch[3], 3, stride=2)
            self.conv2d_5 = TFDWSeparableConvBlock(ch[3], ch[3], 3)
            self.conv2d_6 = TFDWSeparableConvBlock(ch[3], ch[4], 3, stride=2)
            self.conv2d_7 = TFDWSeparableConvBlock(ch[4], ch[4], 3)
            self.conv2d_8 = TFDWSeparableConvBlock(ch[4], ch[4], 3)
            self.conv2d_9 = TFDWSeparableConvBlock(ch[4], ch[4], 3)
            self.conv2d_10 = TFDWSeparableConvBlock(ch[4], ch[4], 3)
            self.conv2d_11 = TFDWSeparableConvBlock(ch[4], ch[4], 3)
            self.conv2d_12 = TFDWSeparableConvBlock(ch[4], ch[5], 3, stride=2)
            self.conv2d_13 = TFDWSeparableConvBlock(ch[5], ch[5], 3)
            self.conv2d_1c = TFLogitsBlock(ch[5], num_classes, 
                                           dropout_keep_prob)
            self.prob = lambda x: prediction_fn(x.reshape(x.shape[0:2]))

