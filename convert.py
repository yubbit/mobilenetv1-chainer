import argparse
import os
import shutil
import tarfile

import chainer
import numpy as np
import tensorflow as tf

import mobilenet_v1_ch
import mobilenet_v1_tf

### ARGPARSE ###
parser = argparse.ArgumentParser(description='Convert Tensorflow MobilenetV1 \
                                              models to Chainer')
parser.add_argument('tgz', help='.tgz file containing the target model')

args = parser.parse_args()

tar = tarfile.open(args.tgz, "r:gz")
model_name = os.path.basename(args.tgz)[:-4]
tar.extractall(model_name)

ckpt_file = os.path.join(model_name, model_name + '.ckpt')
depth_multiplier = float(model_name.split('_')[2])

### TENSORFLOW GRAPH ###
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model_tf, _ = mobilenet_v1_tf.mobilenet_v1(x, num_classes=1001,
                                           depth_multiplier=depth_multiplier,
                                           is_training=False)
saver = tf.train.Saver()

### CHAINER GRAPH ###
model_ch = mobilenet_v1_ch.MobilenetV1(depth_multiplier=depth_multiplier)

### COPY ROUTINE ###
with tf.Session() as sess:
    saver.restore(sess, ckpt_file)

    for var in tf.global_variables():
        if 'MobilenetV1' not in var.name \
          or 'RMSProp' in var.name \
          or 'ExponentialMovingAverage' in var.name:
            continue

        beg = var.name.find('Conv')
        if var.name[beg+8] == '_' or var.name[beg+8] == '/':
            link_name = var.name[beg:beg+8]
        else:
            link_name = var.name[beg:beg+9]
        link_name = link_name.lower()

        link = getattr(model_ch, link_name)

        vals = sess.run(var).astype(np.float32)

        if 'depthwise' in var.name:
            if 'weights' in var.name:
                assert link.dw.W.shape == vals.transpose(3, 2, 0, 1).shape
                link.dw.W = vals.transpose(3, 2, 0, 1)
            elif 'beta' in var.name:
                assert link.dw_bn.beta.shape == vals.shape
                link.dw_bn.beta = vals
            elif 'moving_mean' in var.name:
                assert link.dw_bn.avg_mean.shape == vals.shape
                link.dw_bn.avg_mean = vals
            elif 'moving_variance' in var.name:
                assert link.dw_bn.avg_var.shape == vals.shape
                link.dw_bn.avg_var = vals

        elif 'pointwise' in var.name:
            if 'weights' in var.name:
                assert link.pw.W.shape == vals.transpose(3, 2, 0, 1).shape
                link.pw.W = vals.transpose(3, 2, 0, 1)
            elif 'beta' in var.name:
                assert link.pw_bn.beta.shape == vals.shape
                link.pw_bn.beta = vals
            elif 'moving_mean' in var.name:
                assert link.pw_bn.avg_mean.shape == vals.shape
                link.pw_bn.avg_mean = vals
            elif 'moving_variance' in var.name:
                assert link.pw_bn.avg_var.shape == vals.shape
                link.pw_bn.avg_var = vals

        elif 'Logit' in var.name:
            if 'weights' in var.name:
                assert link.conv.W.shape == vals.transpose(3, 2, 0, 1).shape
                link.conv.W = vals.transpose(3, 2, 0, 1)
            elif 'biases' in var.name:
                assert link.conv.b.shape == vals.shape
                link.conv.b = vals

        else:
            if 'weights' in var.name:
                assert link.conv.W.shape == vals.transpose(3, 2, 0, 1).shape
                link.conv.W = vals.transpose(3, 2, 0, 1)
            elif 'beta' in var.name:
                assert link.bn.beta.shape == vals.shape
                link.bn.beta = vals
            elif 'moving_mean' in var.name:
                assert link.bn.avg_mean.shape == vals.shape
                link.bn.avg_mean = vals
            elif 'moving_variance' in var.name:
                assert link.bn.avg_var.shape == vals.shape
                link.bn.avg_var = vals

### SAVE ROUTINE ###
if not os.path.exists('chainer-models'):
    os.makedirs('chainer-models')
chainer.serializers.save_npz('chainer-models/' + model_name + '.npz', model_ch)

shutil.rmtree(model_name)

