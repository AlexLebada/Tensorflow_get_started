from __future__ import absolute_import, division,print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow._api.v2.feature_column as fc


### SIMPLE LINEAR REGRESSION
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])

#polyfit function - returns an array of coef. that describe the bestfit line of data x,y using least square method
#poly1d function - returns the function based on polyfit returned coeficients
#np.unique - returns sorted x values array,only 1 per each
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
#plt.show()

###LOAD DATASET
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
# .pop - retrieve column 'survived'
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

### EXPLORE DATASET
if 1 == 0:
    print(dftrain.describe())
    print(dftrain.loc[0], y_train.loc[0])

plt.clf()
if 1==0:
    dftrain.age.hist(bins=20)

if 1==0:
    dftrain.sex.value_counts().plot(kind='barh')

if 1==0:
    dftrain['class'].value_counts().plot(kind='barh')

if 1==0:
    pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#plt.show()

### PREPROCESSING DATASET - in this case redefine diff data types into specific TF syntax
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

### CREATE FUNCTION FOR TF OBJECT
def make_input_fn(data_df, label_df, num_epoch=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create TF object when is instantiate it
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epoch)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epoch=1, shuffle=False)

### CREATE MODEL
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

### MODEL TRAINING
linear_est.train(train_input_fn)

result = linear_est.evaluate(eval_input_fn) # get the model metrics by test TF object
#print(result)

### MODEL PREDICTION
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[3])
print(y_eval[3])
print(result[3]['probabilities'][1])
