import tensorflow as tf
from keras import datasets
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

(train_images, train_labes), (test_images, test_labels) = datasets.cifar10.load_data()
