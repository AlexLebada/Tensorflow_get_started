import tensorflow as tf
import keras
import numpy as np
import os
import matplotlib.pyplot as plt


### LOAD DATASET
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#print(train_images.shape)
#print(train_images[0,23,23])
#print(train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if 1==0:
    plt.figure()
    plt.imshow(train_images[5])
    plt.colorbar()
    plt.grid(False)
    plt.show()


###PREPROCESSING
train_image = train_images/255.0
test_images = test_images /255.0

if not os.path.exists('FeedForward_NN_1.keras'):
    ### BUILD MODEL
    #sequential = feedforward NN
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # to 784 array size for dense layers
        keras.layers.Dense(128, activation='relu'), # Input 128. is a hyperparameter. Can be tuned
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ### TRAIN
    model.fit(train_image, train_labels, epochs=10)
    model.save('FeedForward_NN_1.keras')
    ### EVALUATE
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1) # verbose is for progress bar
    print(test_loss, test_acc)
else:
    model = keras.models.load_model('FeedForward_NN_1.keras')

COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
