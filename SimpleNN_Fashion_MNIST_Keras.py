
# coding: utf-8

# In[1]:


#A simple implementation of neural network classification using Keras
#Source: https://www.tensorflow.org/tutorials/keras/basic_classification

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[2]:


#import the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


#store the image class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[4]:


#Explore the Data


# In[5]:


train_images.shape


# In[6]:


len(train_labels)


# In[9]:


test_images.shape


# In[10]:


len(test_labels)


# In[11]:


#Preprocess the data


# In[12]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)


# In[13]:


#need to scale the values to a range 0 to 1
train_images = train_images / 255.0

test_images = test_images / 255.0


# In[14]:



#Display the first 25 images from the training set and display the class name below each image.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[15]:


#Simple NN: Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[16]:


model.summary()


# In[17]:


#Compile the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[18]:


#Train the model
model.fit(train_images, train_labels, epochs=5)


# In[20]:


#Evaluate the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[65]:


#Make Predictions
predictions = model.predict(test_images)
predictions[0]


# In[66]:


#label with the highest confidence value
np.argmax(predictions[0])


# In[67]:


#check the test label to see if it is correct
test_labels[0]

