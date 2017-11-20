"""Copyright 2017 Neosapience, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import numpy as np
import os

from mnist_deep import deepnn
import tensorflow as tf
import matplotlib.pyplot as plt
from load_mnist import load_mnist
import darkon
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
FLAGS = None
SEED = 75
import input_data
class MyFeeder(darkon.InfluenceFeeder):
    def __init__(self, train_dataset, test_dataset):
        self.test_data = test_dataset.images
        self.test_label = test_dataset.labels
        self.train_data = train_dataset.images
        self.train_label = train_dataset.labels
        self.train_dataset = train_dataset

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        return self.train_dataset.next_batch(batch_size, shuffle=False)

    def train_one(self, idx):
        return self.train_data[idx], self.train_label[idx]

#    def reset(self):
#np.random.seed(SEED)


class InfluenceInspectTest():
    def influence(self):
#        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        mnist = load_mnist(FLAGS.data_dir, subsample_rate=FLAGS.subsample_rate,
                random_seed=FLAGS.random_seed)
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        y_conv, keep_prob = deepnn(x)
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()

        ckpt_path = FLAGS.ckpt_path
        print('load checkpoint: {}'.format(ckpt_path))
        saver.restore(sess, ckpt_path)
        workspace = FLAGS.work_dir

        print 'Start influence calculation...'
        print '----------------------------'

        inspector = darkon.Influence(
            workspace=workspace,
            feeder=MyFeeder(mnist.train, mnist.test),
            loss_op_train=cross_entropy,
            loss_op_test=cross_entropy,
            x_placeholder=x,
            y_placeholder=y_,
            feed_options={keep_prob: 1.0}
        )
        test_indices=range(mnist.test.num_examples) # all test samples
        train_indices=range(mnist.train.num_examples) # all train samples
        up_inf = inspector.upweighting_influence(
                    sess,
                    test_indices=test_indices,
                    test_batch_size=100,
                    approx_params={'scale': 1e4,
                                    'damping': 0.01,
                                    'num_repeats': 1,
                                    'recursion_batch_size': 100,
                                    'recursion_depth': 10000},
                    train_indices=train_indices,
                    num_total_train_example=mnist.train.num_examples,
                    force_refresh=True)

        np.savetxt(os.path.join(workspace, 'up_inf.txt'), up_inf)
#        up_inf = np.loadtxt(os.path.join(workspace, 'up_inf.txt'))
        sorted_indices = np.argsort(up_inf)
        Nx = 10
        Ny = 10
        N = Nx*Ny
        img_negative = np.zeros((Ny*28, Nx*28))
        img_positive = np.zeros((Ny*28, Nx*28))
        n_worst_indices = sorted_indices[:N]
        n_best_indices = sorted_indices[-N:][::-1]
        n_worst_indices = n_worst_indices[np.where(up_inf[n_worst_indices]<0)[0]]
        n_best_indices = n_best_indices[np.where(up_inf[n_best_indices]>0)[0]]

        print('\nN-Worst influence:')
        for idx in n_worst_indices:
            print('[{}] {}'.format(idx, up_inf[idx]))
        print('\nN-Best influence:')
        for idx in n_best_indices:
            print('[{}] {}'.format(idx, up_inf[idx]))
        np.savetxt(os.path.join(workspace, 'n_worst_idx.txt'), n_worst_indices)
        np.savetxt(os.path.join(workspace, 'n_best_idx.txt'), n_best_indices)

        i=0
        for y in range(Ny):
            for x in range(Nx):
                if i < len(n_worst_indices):
                    img_negative[y*28:y*28+28, x*28:x*28+28] = np.reshape(
                        mnist.train.images[train_indices[n_worst_indices[i]]], [28,28])
                if i < len(n_best_indices):
                    img_positive[y*28:y*28+28, x*28:x*28+28] = np.reshape(
                        mnist.train.images[train_indices[n_best_indices[i]]], [28,28])
                i+=1

        plt.imshow(img_negative)
        plt.axis('off')
        plt.axis('tight')
        plt.savefig(os.path.join(workspace,'img_neg.png'))
        plt.imshow(img_positive)
        plt.axis('off')
        plt.axis('tight')
        plt.savefig(os.path.join(workspace,'img_pos.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./tmp',
                        help='directory for storing input data')
    parser.add_argument('--work_dir', type=str,
                        default='./origin',
                        help='directory for storing input data')
    parser.add_argument('--ckpt_path', type=str,
                        default='./origin/model.ckpt-19999',
                        help='trained  model')
    parser.add_argument('--subsample_rate', type=int,
                        default=1,
                        help='data subsample rate')
    parser.add_argument('--random_seed', type=int,
                        default=SEED,
                        help='random seed')

    FLAGS, unparsed = parser.parse_known_args()
    a = InfluenceInspectTest()
    a.influence()
