import chainer
import numpy as np
import tensorflow as tf

import chainer.functions as F
import chainer.links as L

from chainercv.links.model.pickable_sequential_chain import PickableSequentialChain

import mobilenet_v1_tf

## CHAINER ##
class Graph(PickableSequentialChain):
    def __init__(self):
        super(Graph, self).__init__()
        with self.init_scope():
            pd = ((0, 0), (0, 0), (0, 1), (0, 1))
            self.pad_0 = lambda x: F.pad(x, pd, mode='constant', constant_values=0)
            self.conv_0 = L.Convolution2D(3, 32, 3, 2, 0, True)
            self.bn_0 = L.BatchNormalization(32, decay=0.9997, eps=0.001)
#            self.relu_0 = F.relu
#            self.conv_1_dw = L.DepthwiseConvolution2D(32, 1, 3, 1, 1, nobias=True)

model_ch = Graph()

## TENSORFLOW ##
x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model_tf = mobilenet_v1_tf.mobilenet_v1(x, num_classes=1001, is_training=False)

saver = tf.train.Saver()

## RUN ##
img_tf = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)
#img_tf = np.ones((1, 224, 224, 3)).astype(np.float32)
img_ch = img_tf.transpose(0, 3, 1, 2)

with tf.Session() as sess:
    saver.restore(sess, 'mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt')

    ### DONE ###
    #node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_0/Conv2D:0')
    #val_tf = sess.run(node_tf, feed_dict={x: img_tf})

    w_node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Conv2d_0/weights:0')
    w_tf = sess.run(w_node_tf)
    w_ch = w_tf.transpose(3, 2, 0, 1).astype(np.float32)

    model_ch.conv_0.W = w_ch

    node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm:0')
    val_tf = sess.run(node_tf, feed_dict={x: img_tf})

    beta_node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Conv2d_0/BatchNorm/beta:0')
    beta_tf = sess.run(beta_node_tf)
    mean_node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Conv2d_0/BatchNorm/moving_mean:0')
    mean_tf = sess.run(mean_node_tf)
    var_node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Conv2d_0/BatchNorm/moving_variance:0')
    var_tf = sess.run(var_node_tf)
    
    model_ch.bn_0.beta = beta_tf
    model_ch.bn_0.avg_mean = mean_tf
    model_ch.bn_0.avg_var = var_tf

    #node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_0/Relu:0')
    #val_tf = sess.run(node_tf, feed_dict={x: img_tf})

    #node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise:0')
    #val_tf = sess.run(node_tf, feed_dict={x: img_tf})

    #w_node_tf = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Conv2d_1_depthwise/depthwise_weights:0')
    #w_tf = sess.run(w_node_tf)
    #w_ch = w_tf.transpose(3, 2, 0, 1).astype(np.float32)

    #model_ch.conv_1_dw.W = w_ch

    ### DOING ###


with chainer.using_config('train', False):
    val_ch = model_ch(img_ch)
val_ch = val_ch.data
val_ch = val_ch.transpose(0, 2, 3, 1)

print(val_tf.shape, val_ch.shape)
print(val_tf[0, 0:5, 0:5, 0])
print(val_ch[0, 0:5, 0:5, 0])
print()
print(val_tf[0, -5:, -5:, 0])
print(val_ch[0, -5:, -5:, 0])

'''
### NOTES ###
- Both tensorflow and chainer use zero padding for convolution. For tensorflow,
  if the necessary padding to generate a SAME output is an odd number, it will
  allocate the odd portion to the end of the indices (e.g. bottom right of an
  image). Chainer will apply this to the beginning of the indices (e.g. top
  left). Tensorflow's behavior can be replicated in chainer by applying the
  built-in padding function and then applying a convolutional link with 0
  padding
'''
