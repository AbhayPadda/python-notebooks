
# coding: utf-8

# # 'Hello-world' for Image Classification
# 
# ## Simple Convolutional Neural Network Model for classifying images in MNIST dataset

# In[1]:


## Loading libraries required for this
import matplotlib.pyplot as plt
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# ### Define simple C-NN model
# 
# #### Convolutional neural networks are more complex than standard multi-layer perceptrons, so we will start by using a simple structure to begin with that uses all of the elements for state of the art results. Below summarizes the network architecture.
# 
# #### 1. The first hidden layer is a convolutional layer called a Convolution2D. The layer has 32 feature maps, which with the size of 5×5 and a rectifier activation function. This is the input layer, expecting images with the structure outline above [pixels][width][height].
# #### 2. Next we define a pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
# #### 3. The next layer is a regularization layer using dropout called Dropout. It is configured to randomly exclude 20% of neurons in the layer in order to reduce overfitting.
# #### 4. Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to be processed by standard fully connected layers.
# #### 5. Next a fully connected layer with 128 neurons and rectifier activation function.
# #### 6. Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like predictions for each class.
# 
# #### As before, the model is trained using logarithmic loss and the ADAM gradient descent algorithm.

# In[2]:


def cnn_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# ### Reshape dataset and normalize the input params

# In[6]:


## Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Convert output variable to categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Normalizing the data
X_train = X_train / 255
X_test = X_test / 255

num_classes = y_train.shape[1]


# ### Build the model, then run it and evaluate the results

# In[7]:


# build the model
model = cnn_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


# #### The baseline error rate for this is 1.03% which is not bad considering that we have only created a Convolutional Neural Network model. The error rate has decreased if we compare this with the simple neural network model where the error rate was 1.77%. We can further reduce the error by creating much larger and complex C-NN model.
