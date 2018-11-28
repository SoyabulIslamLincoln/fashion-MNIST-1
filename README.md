# Fashion-MNIST with Keras 
Building a neural network classifier using Fashion MNIST dataset and Tensorflow Keras.

Many thanks to the following tutorials, which I used for reference and guidance:

[Train your first neural network: basic classification](https://www.tensorflow.org/tutorials/keras/basic_classification)

[LeNet: Convolutional Neural Network in Python](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)

# The Background

## Original MNIST
The original [MNIST](http://yann.lecun.com/exdb/mnist/) is a dataset of handwritten digits which contains 60,000 training images and 10,000 test images, each of size 28x28. MNIST is widely regarded as the "hello world" of deep learning and is often the starting place for many people who are new to deep learning to start building their first neural networks.

<img src ="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" align="center">

*MNIST Visualization [Source: Wikipedia](https://en.wikipedia.org/wiki/MNIST_database).*


## Fashion MNIST

More recently, researchers at [Zalando](www.zalando.com) have developed a dataset [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), intended to be a drop-in replacement for the original MNIST.

<img src="https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png" width="600" align="center">

*Fashion MNIST visualization (by [Zalando](https://github.com/zalandoresearch/fashion-mnist), MIT License).*


This dataset also contains 60,000 training images and 10,000 test images, and has similar structure in train/test split as the original MNIST. Here, the 10 classes correspond to 10 different articles of clothing.

**Why Fashion MNIST?** The Zolando researchers argue the original MNIST is too easy (i.e. to achieve high accuracy), overused, and does not sufficiently represent modern computer vision tasks. Fashion MNIST is proposed as a relatively more challenging dataset than the original. Yet, it is still a good starting point for introduction to deep learning.

So! - here starts my adventure in building a neural network classifier using Fashion MNIST. 

- First, I build a simple neural network using 3 fully-connected layers 
- Then, I build a slightly larger Convolutional Neural Network with Dropout Regularization.

**Goal:** To show how the training and test accuracy improves with the neural network optimizations.


# Simple Neural Network
```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
```

First, we load the dataset
```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
The class names are not included in the dataset so we store them here:
```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

We'll do a little bit of insights into the data set. There are 60,000 images in the training set, each is 28x28 pixels:
```python
train_images.shape
(60000, 28, 28) 
```
There are 60,000 labels in the training set, and each label is an integer between 0 and 9:

```python
len(train_labels)
```
```
60000
```
There are 10,000 images in the test set, each is 28x28 pixels:
```python
test_images.shape
```
```
(10000, 28, 28)
```
And finally, there are 10,000 labels in the test set:
```python
len(test_labels)
```
```
10000
```

We need to preprocess the data; we scale the images to a range of 0 to 1 by dividing by 255 (range of pixel values): 

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

Here is where we build the model for a Simple NN with 3 layers 

1st layer: "unrolls" the images from a 2d-array of 28 by 28 pixels to a 1d-array of 28 * 28 = 784 pixels

2nd layer: Fully-connected layer. Has 128 nodes.

3rd layer: 10-node softmax layer, which returns an array of 10 probability scores that sum to 1. 
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

[WIP]
