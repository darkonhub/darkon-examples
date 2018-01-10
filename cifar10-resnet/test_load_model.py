from cifar10_train import Train
from cifar10_input import *

import tensorflow as tf
import numpy as np

import darkon


maybe_download_and_extract()

# tf model checkpoint
check_point = 'pre-trained/model.ckpt-79999'

net = Train()
net.build_train_validation_graph()
saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    saver.restore(sess, check_point)
    tf.summary.FileWriter('./workspace', sess.graph)

