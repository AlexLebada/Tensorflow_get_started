import tensorflow as tf
import os
from keras import datasets, layers, models
import keras
import numpy as np
import matplotlib.pyplot as plt

### LOAD
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

### PREPROCESS
train_images, test_images = train_images/255.0, test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

if 1==0:
    plt.figure()
    plt.imshow(train_images[5])
    plt.colorbar()
    plt.grid(False)
    plt.show()

if not os.path.exists('Conv_NN_1.keras'):
    ### BUILD MODEL
    model = models.Sequential() # declaration
    model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32, 32, 3))) # this layer takes the input image and outputs feature map
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()

    ### TRAINING MODEL
    # logits=unnormalized output values from last layer, and if True, internally values are transformed into probabilities(softmax)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # by giving validation_data in this phase, it can do an early stopping if model overfit
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save('Conv_NN_1.keras')
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nLoss: ', test_loss,'\nAccuracy: ', test_acc)
else:
    model = keras.models.load_model('Conv_NN_1.keras')
    ### PREDICT
    prediction = model.predict(test_images)
    plt.figure()
    plt.imshow(test_images[5])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print(prediction[5])
    print(test_labels[5])