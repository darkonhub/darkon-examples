# darkon-examples
Examples and use cases for [darkon](https://github.com/darkonhub/darkon)

## Install python package
pip install darkon

## mnist-cnn
Influence score calculation example
It first trains CNN model for MNIST data, and calculate influence score to find most helpful and harmful training samples.
Then, retrain the model after removing 100 most helpful training samples and 100 most harmful training samples, respectively.

## With cifar10-resnet
* dataset: [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* network: [resnet110](https://github.com/wenxinxu/resnet-in-tensorflow)

### Example for upweight influence function
* [example code](https://github.com/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet.ipynb)
* [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet.ipynb)

### Example for mislabel with all of layers
* [example code](https://github.com/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_all_layers.ipynb)
* [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_all_layers.ipynb)

### Example for mislabel with one top layer
* [example code](https://github.com/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_one_layer.ipynb)
* [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_one_layer.ipynb)


