import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os
from math import ceil

def create_net(input, keep_prob, nr_classes, name):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = tf.nn.relu):
        net = slim.conv2d(input, 32, [5,5], padding='SAME', scope=name +"/convolutional_part/conv1")
        net = slim.max_pool2d(net, [2, 2], [2, 2], scope=name + "/convolutional_part/pool1")
        net = slim.conv2d(net, 64, [5,5], padding='SAME', scope=name + "/convolutional_part/conv2")
        net = slim.max_pool2d(net, [2, 2], [2, 2], scope=name + "/convolutional_part/pool2")

        net = slim.flatten(net, scope=name + "/flatten")
        net = slim.fully_connected(net, 1024, scope=name + "/fc_1")
        
        net = slim.dropout(net, keep_prob, scope=name + "/dropout_1")
        return slim.fully_connected(net, nr_classes, scope=name + "/fc_out")


def create_missing_dirs(relative_paths):
    for path in relative_paths:
        path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            os.makedirs(path)

batch_size = 100
max_epochs = 10
dataset_path = r"data/fashion"
model_path = 'checkpoints'
log_path = 'logs'
create_missing_dirs([model_path, log_path])
mnist = input_data.read_data_sets(dataset_path, one_hot = True)
steps_per_epoch = ceil(len(mnist.train.images) / batch_size)
val_steps = ceil(len(mnist.validation.images)/batch_size)
test_steps = ceil(len(mnist.test.images)/batch_size)

#Declare input variables
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

y = create_net(x_img, keep_prob, 10, "testnet")

#Declare losses
cross_entropy = tf.losses.softmax_cross_entropy(y_, y)
tf.summary.scalar('cross_entropy', cross_entropy)

#Declare metrics
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

#setup trainer
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(1e-4).minimize(tf.losses.get_total_loss(True), global_step=global_step)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    checkpoint = tf.train.get_checkpoint_state(model_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")


    writer = tf.summary.FileWriter(log_path, sess.graph)
    iteration_cnt = 0
    for epoch in range(max_epochs):
        for i in range(steps_per_epoch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)            
            #train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            _, train_loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            print("Step " + str(i) + "/" + str(steps_per_epoch)+ ": " + str(train_loss), end='\r')
            if i % 50 ==0:
                merged = tf.summary.merge_all()
                summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
                writer.add_summary(summary, iteration_cnt * batch_size)
            iteration_cnt += 1

        accuracy_agg = 0
        for i in range(val_steps):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)
            accuracy_agg += accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        saver.save(sess, os.path.join(model_path, "testnet"), global_step=epoch)
        print('Epoch %d, validation accuracy %g' % (epoch, 1.0 * accuracy_agg / val_steps))

    accuracy_agg = 0.0
    for i in range(test_steps):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        accuracy_agg += accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    print('Final test accuracy %g' % (1.0 * accuracy_agg / test_steps))
    merged = tf.summary.merge_all()
    summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    writer.add_summary(summary, iteration_cnt * batch_size)