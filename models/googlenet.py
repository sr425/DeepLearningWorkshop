import tensorflow as tf
import tensorflow.contrib.slim as slim
from . import tf_util

#From the paper "Going Deeper with Convolutions" Szegedy et al.
def build_googlenet(x):
    with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(x, 64, [7, 7], [2, 2], scope="conv_1")
        net = slim.max_pool2d(net, [3, 3], [2, 2], padding="SAME", scope="maxpool_1")
        net = slim.batch_norm(net, fused=True, scope="BN_1")
        net = slim.conv2d(net, 64, [1, 1], activation_fn=tf.nn.relu, scope="conv_1_reduction")

        net = slim.conv2d(net, 192, [3, 3], scope="conv_2")
        net = slim.max_pool2d(net, [3, 3], [2, 2], padding="SAME", scope="maxpool_2")
        net = slim.batch_norm(net, fused=True, scope="BN_2")

        net = tf_util.inception_layer(net, [96, 16], [64, 128, 32, 32], "inception_3a")
        net_3 = tf_util.inception_layer(net, [128, 32], [128, 192, 96, 64], "inception_3b")

        net = slim.max_pool2d(net_3, [3, 3], [2, 2], padding="SAME", scope="maxpool_3")

        net = tf_util.inception_layer(net, [96, 16], [192, 208, 48, 64], "inception_4a")
        net = tf_util.inception_layer(net, [112, 24], [160, 224, 64, 64], "inception_4b")
        net = tf_util.inception_layer(net, [128, 24], [128, 256, 64, 64], "inception_4c")
        net = tf_util.inception_layer(net, [144, 32], [112, 288, 64, 64], "inception_4d")
        net_4 = tf_util.inception_layer(net, [160, 32], [256, 320, 128, 128], "inception_4e")

        net = slim.max_pool2d(net_4, [3, 3], [2, 2], padding="SAME", scope="maxpool_4")

        net = tf_util.inception_layer(net, [160, 32], [256, 320, 128, 128], "inception_5a")
        net = tf_util.inception_layer(net, [192, 48], [384, 384, 128, 128], "inception_5b")
    return net, net_3, net_4

def build_classification(x, output_channels):
    net, _, _ = build_googlenet(x)
    net = slim.avg_pool2d(net, [7, 7], [1, 1], scope="avg_pool_7x7")
    net = slim.dropout(net, 0.4, scope="dropout_1")
    return slim.fully_connected(net, output_channels, scope="final")
