from __future__ import absolute_import, division,print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# .utils.get_file - download from URL
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train_y = train.pop('Species')
test_y = test.pop('Species')

def input_fn(features, labels, training=True, batch_size=256):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices(dict(features), labels)
        if training:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat()
        return dataset
    return input_function

train_input_fn = input_fn(train, train_y)
eval_input_fn = input_fn(test, test_y, training=False)

my_feature_columns = []
# keys are the columns of the dictionary/dataframe = equivalent to .columns
for key in train.columns:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#print(my_feature_columns)

###INSTANTIATE/BUILD THE MODEL
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10], # 2 layers
    n_classes=3)

### TRAIN
# in this case lambda is a way to delay accessing data batches by the model, when needed
# steps are the rows in this case, instead of epochs
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

### EVALUATE
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

### PREDICTION
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))
