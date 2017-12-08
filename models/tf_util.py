import tensorflow as tf
import tensorflow.contrib.slim as slim

def inception_layer(x, reductions, out_depths, name=None, concat_axis=3):
    conv1x1 = slim.conv2d(x, out_depths[0], [1, 1], activation_fn=tf.nn.relu, scope=name + "/conv_1x1")
    reduction_3x3 = slim.conv2d(x, reductions[0], [1, 1], activation_fn=tf.nn.relu, scope=name + "/conv_3x3_reduce")
    reduction_5x5 = slim.conv2d(x, reductions[1], [1, 1], activation_fn=tf.nn.relu, scope=name + "/conv_5x5_reduce")
    max_pool_1 = slim.max_pool2d(x, [3, 3], [1, 1], padding="SAME", scope=name + "/maxpool")

    conv_3x3 = slim.conv2d(reduction_3x3, out_depths[1], [3, 3], activation_fn=tf.nn.relu, scope=name + "/conv_3x3")
    conv_5x5 = slim.conv2d(reduction_5x5, out_depths[2], [5, 5], activation_fn=tf.nn.relu, scope=name + "/conv_5x5")
    conv_1x1_pool = slim.conv2d(max_pool_1, out_depths[3], [1, 1], activation_fn=tf.nn.relu, scope=name + "/conv_pool_proj")

    return tf.concat([conv1x1, conv_3x3, conv_5x5, conv_1x1_pool], axis=concat_axis, name=name + "/concat")
