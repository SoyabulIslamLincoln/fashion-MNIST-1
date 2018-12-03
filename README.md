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

- First, I build a simple 3 layer neural network with fully connected layers 
- Then, I build a slightly larger Convolutional Neural Network

**Goal:** To show how the training and test accuracy improves with a more optimized neural network architecture.


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

We need to preprocess the data; we scale the images to a range of 0 to 1 by dividing by 255 (image range of pixel values): 

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

Here is where we build the model for a Simple NN with 3 layers 
 
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```
We can see some details about the model using model.summary()
```python
model.summary()
```
```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_9 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_18 (Dense)             (None, 128)               100480    
_________________________________________________________________
dense_19 (Dense)             (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```

1st layer: "unrolls" the images from a 2d-array of 28 by 28 pixels to a 1d-array of 28 * 28 = 784 pixels

2nd layer: Fully-connected layer. Has 128 nodes.

3rd layer: 10-node softmax layer, which returns an array of 10 probability scores (for the 10 classes) that sum to 1.

We then compile the model using Adam optimizer:
```python
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

Finally, we train the model:
```python
model.fit(train_images, train_labels, epochs=5)
```
```python
Epoch 1/5
60000/60000 [==============================] - 5s 75us/step - loss: 0.4993 - acc: 0.8247
Epoch 2/5
60000/60000 [==============================] - 4s 72us/step - loss: 0.3765 - acc: 0.8649
Epoch 3/5
60000/60000 [==============================] - 4s 71us/step - loss: 0.3379 - acc: 0.8766
Epoch 4/5
60000/60000 [==============================] - 4s 71us/step - loss: 0.3129 - acc: 0.8849
Epoch 5/5
60000/60000 [==============================] - 5s 78us/step - loss: 0.2954 - acc: 0.8902
```

Then evaluate the accuracy:
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
```python
print('Test accuracy:', test_acc)
```
```python
10000/10000 [==============================] - 0s 39us/step
Test accuracy: 0.8732
```

We get to a **train accuracy of 89.02%** and a **test accuracy of 87.32%**, not bad for a 3 layer network!

But can we do better? 

# Convolutional Neural Network (CNN)
Convolutional Neural Networks are excellent NN architectures to apply to computer vision tasks. CNNs have been proven to be very effective effective for image recognition, object recognition, and even natural language processing tasks. An excellent  description on CNNs can be found in [Stanford's CS231n course notes:](http://cs231n.github.io/convolutional-networks/)

For this example, we use the same data set from the simple neural network example above, but we build a different model; in particular, this CNN model is based on the LeNet architecture, first introduced by Yann LeCun et a. in the paper [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). 


<img src ="https://www.pyimagesearch.com/wp-content/uploads/2016/06/lenet_architecture-768x226.png" width="600" align="center"> 


*LeNet Architecture Visualization. [Source](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python).*

 
Let's build the model:
```python
model = keras.models.Sequential([
    keras.layers.Conv2D(20, (5,5), padding='same', activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(50, (5,5), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

We can see some details about the model:
```python
model.summary()
```
```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 28, 28, 20)        520       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 20)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 50)        25050     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 7, 50)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2450)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 500)               1225500   
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5010      
=================================================================
Total params: 1,256,080
Trainable params: 1,256,080
Non-trainable params: 0
_________________________________________________________________
```
It is important to note that the first layer has an input shape argument of (28, 28, 1). This is because the first convolutional layer expects each input example to be a 28x28 image, each with 1 channel (because the images are greyscale; if they were color it woudl be 3 for RGB).  However, the shape of our training set is (60000, 28, 28) as we saw in the simple NN example. Thus, we need to reshape our train and test images to match what the model expects:

```python
#reshape the model
train_images=train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images=test_images.reshape(test_images.shape[0], 28, 28 ,1) 
```

The final layer of the CNN model will output a vector of size 10 for each label. Each element of the vector is a "probability" that indicates the confidence of the model to assign that input example to a particular class. However, our train and test labels as defined in the simple neural network example are not vectors but rather integers, which we need to convert to vectors in what is cahlled "one hot" encoding.

With one hot encoding, each label is represented by a vector, and each element of the vector will be of value 0 or 1 depending on what class the label corresponds to. (e.g. [0,0,0,1,0,0,0,0,0,0] would be class #3).

```python
#convert current representation of the labels to "One Hot Representation"
train_labels=keras.utils.to_categorical(train_labels)
test_labels=keras.utils.to_categorical(test_labels)
```

```python
#display the new train and test dimensions
print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)
print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)
```
```python
train_images shape: (60000, 28, 28, 1)
test_images shape: (10000, 28, 28, 1)
train_labels shape: (60000, 10)
test_labels shape: (10000, 10)
```

Now, we are read to compile the model. 
```python
#Compile the model using Adam Optimizer and categorical crossentropy loss
model.compile(optimizer=tf.train.AdamOptimizer(), 
             loss='categorical_crossentropy',
              metrics=['accuracy'])
```

And finally, let's train!

```python
model.fit(train_images, train_labels, epochs=5)
```

```python
Epoch 1/5
60000/60000 [==============================] - 119s 2ms/step - loss: 0.3882 - acc: 0.8598
Epoch 2/5
60000/60000 [==============================] - 118s 2ms/step - loss: 0.2579 - acc: 0.9047
Epoch 3/5
60000/60000 [==============================] - 118s 2ms/step - loss: 0.2128 - acc: 0.9200
Epoch 4/5
60000/60000 [==============================] - 121s 2ms/step - loss: 0.1791 - acc: 0.9327
Epoch 5/5
60000/60000 [==============================] - 119s 2ms/step - loss: 0.1521 - acc: 0.9428
```
Let's evaluate the accuracy against the test images:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
```python
10000/10000 [==============================] - 7s 725us/step
Test accuracy: 0.918
```

With the CNN model, we get to a **train accuracy of 94.28%** and a **test accuracy of 91.8%**! 

From this we can see that implementing a CNN model allowed us to increase the train and test accuracy in our Fashion MNIST classifier. Optimization of the neural network architecture is a powerful way to improve the performance of your algorithm. 

I will caveat that in this case, there are still other opportunities for improvements. Maybe running the training for longer or getting a bigger network might improve the train accuracy. Another observation is that the test accuracy is lower than the train accuracy, which could indicate a high variance issue. Maybe adding regularization techniques such as Dropout might help as well. 

In any case, this example is a good demonstration of how to build and train simple and convolutional neural networks :)
