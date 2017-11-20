#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from mnist import *
import numpy as np
from tensorflow.python.framework import dtypes

def load_mnist(train_dir, subsample_rate=1, random_seed=0):
    np.random.seed(random_seed)
    mnist = read_data_sets(train_dir, one_hot=True)
    train_images = 255.*mnist.train.images
    train_labels = mnist.train.labels
    perm = np.random.permutation(mnist.train.num_examples)
    num_subsample = int(mnist.train.num_examples/subsample_rate)
    perm = perm[:num_subsample]
    train_images = train_images[perm,:]
    train_labels = train_labels[perm,:]

    val_images = 255.*mnist.validation.images
    val_labels = mnist.validation.labels

    test_images = 255.*mnist.test.images
    test_labels = mnist.test.labels
    options = dict(reshape=False, seed=None)
    train = DataSet(train_images, train_labels, **options)
    val = DataSet(val_images, val_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=val, test=test)

def remove_samples(dataset, remove_idx=[]):
    train_images = 255.*dataset.train.images
    train_labels = dataset.train.labels
    train_images = np.delete(train_images, remove_idx, axis=0)
    train_labels = np.delete(train_labels, remove_idx, axis=0)
    options = dict(reshape=False, seed=None)
    train = DataSet(train_images, train_labels, **options)
    val = DataSet(255.*dataset.validation.images, dataset.validation.labels, **options)
    test = DataSet(255.*dataset.test.images, dataset.test.labels, **options)
    return base.Datasets(train=train, validation=val, test=test)

