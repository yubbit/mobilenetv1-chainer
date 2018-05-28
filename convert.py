import chainer
import numpy as np
import tensorflow as tf

import mobilenet_v1_ch
import mobilenet_v1_tf

### TENSORFLOW GRAPH ###
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model_tf, _ = mobilenet_v1_tf.mobilenet_v1(x, num_classes=1001,
                                           is_training=False)
saver = tf.train.Saver()

### CHAINER GRAPH ###
model_ch = mobilenet_v1_ch.MobilenetV1()

### COPY ROUTINE ###
with tf.Session() as sess:
    saver.restore(sess, 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt')

    writer = tf.summary.FileWriter('logs', sess.graph)
    writer.close()

    for var in tf.global_variables():
        if 'MobileNetV1' not in var.name \
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

        if 'wise' in var.name:
            if 'depth' in var.name:
                if 'weights' in var.name:
                    link.dw.W = vals.transpose(3, 2, 0, 1)
                elif 'beta' in var.name:
                    link.dw_bn.beta = vals
                elif 'moving_mean' in var.name:
                    link.dw_bn.avg_mean = vals
                elif 'moving_variance' in var.name:
                    link.dw_bn.avg_var = vals
            elif 'point' in var.name:
                if 'weights' in var.name:
                    link.pw.W = vals.transpose(3, 2, 0, 1)
                elif 'beta' in var.name:
                    link.pw_bn.beta = vals
                elif 'moving_mean' in var.name:
                    link.pw_bn.avg_mean = vals
                elif 'moving_variance' in var.name:
                    link.pw_bn.avg_var = vals
        elif 'Logit' in var.name:
            if 'weights' in var.name:
                link.conv.W = vals.transpose(3, 2, 0, 1)
            elif 'biases' in var.name:
                link.conv.b = vals
        else:
            if 'weights' in var.name:
                link.conv.W = vals.transpose(3, 2, 0, 1)
            elif 'beta' in var.name:
                link.bn.beta = vals
            elif 'moving_mean' in var.name:
                link.bn.avg_mean = vals
            elif 'moving_variance' in var.name:
                link.bn.avg_var = vals

img_tf = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)
img_tf = img_tf / 255
img_tf = img_tf - 0.5
img_tf = img_tf * 2
img_ch = img_tf.transpose(0, 3, 1, 2)

with tf.Session() as sess:
    saver.restore(sess, 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt')
    node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_0/Relu:0')
    val_tf = sess.run(node_tf, feed_dict={x:img_tf})

with chainer.using_config('train', True):
    model_ch.pick = 'conv2d_0'
    val_ch = model_ch(img_ch).data

val_ch = val_ch.transpose(0, 2, 3, 1)
print(val_tf.shape, val_ch.shape)
print(val_tf[0, 0:5, 0:5, 0])
print(val_ch[0, 0:5, 0:5, 0])
print()
print(val_tf[0, -5:, -5:, 0])
print(val_tf[0, -5:, -5:, 0])
