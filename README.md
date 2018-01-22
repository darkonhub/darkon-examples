# darkon-examples
Examples and use cases for [darkon](https://github.com/darkonhub/darkon)

## Install python package
pip install darkon

## darkon.Influence

### mnist, cnn
Influence score calculation example
It first trains CNN model for MNIST data, and calculate influence score to find most helpful and harmful training samples.
Then, retrain the model after removing 100 most helpful training samples and 100 most harmful training samples, respectively.
* [code](https://github.com/darkonhub/darkon-examples/blob/master/mnist)

### cifar10, resnet
* dataset: [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* network: [resnet110](https://github.com/wenxinxu/resnet-in-tensorflow)
* Upweight influence function
  * [example code](https://github.com/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet.ipynb)
  * [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet.ipynb)
* Mislabel detection with all of layers
  * [example code](https://github.com/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_all_layers.ipynb)
  * [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_all_layers.ipynb)
* Mislabel detection with one top layer
  * [example code](https://github.com/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_one_layer.ipynb)
  * [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/cifar10-resnet/influence_cifar10_resnet_mislabel_one_layer.ipynb)


## darkon.Gradcam

### ImageNet, resnet
* model: [used pre-trained model in TF slim](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
* network: [resnet v1 50 in TF slim](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)
* Gradcam & Guided Gradcam
  * [example code](https://github.com/darkonhub/darkon-examples/blob/master/gradcam/GradcamDemo.ipynb)
  * [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/gradcam/GradcamDemo.ipynb)

### Sentence polarity dataset, Text sentiment classification by CNN
* model: [used pre-trained model](https://raw.githubusercontent.com/darkonhub/darkon-examples/master/gradcam/sequence.tar)
* network: [cnn text classification by dennybritz](https://github.com/dennybritz/cnn-text-classification-tf)
* Gradcam for text sentiment classification
  * [example code](https://github.com/darkonhub/darkon-examples/blob/master/gradcam/GradcamDemoSequence.ipynb)
  * [html view](http://nbviewer.jupyter.org/github/darkonhub/darkon-examples/blob/master/gradcam/GradcamDemoSequence.ipynb)
