"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

COLUMNS = ["Name", "Sex", "Pclass", "Embarked", "Cabin", "Age", "SibSp", "Parch", "Fare"]
LABEL_COLUMN = "Survived"
CATEGORICAL_COLUMNS = ["Sex", "Embarked", "Cabin", "Name"]
CONTINUOUS_COLUMNS = ["Age", "Pclass", "SibSp", "Parch", "Fare"]

model_dir = tempfile.NamedTemporaryFile(delete=True).name # or "./data/model"

def read_data_and_return_train_test(split=0.2): # 80% train, 20% validate
  print("Splitting with test data: "+ str(split*100) + "%")
  train = pd.read_csv("data/train.csv")

  # some ages are NaN set the median age for those
  train['Age'].fillna(train['Age'].median(), inplace=True)

  if split == 0: # don't split into train, train_test they should be equal
    train, test = train, train
  else:
    train, test = train_test_split(train, test_size = split, random_state=42)
  print("data train:"+ str(train.shape) + " test:"+ str(test.shape))

  return train, test

def build_estimator():
  """Build an estimator."""
  
  # sparse base columns.
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="Sex", keys=["female", "male"])
  embarked = tf.contrib.layers.sparse_column_with_keys(column_name="Embarked", keys=["C", "Q", "S"]) # port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
  name = tf.contrib.layers.sparse_column_with_hash_bucket("Name", hash_bucket_size=1000) # cabin number
  cabin = tf.contrib.layers.sparse_column_with_hash_bucket("Cabin", hash_bucket_size=1000) # cabin number

  # continuous base columns.
  age = tf.contrib.layers.real_valued_column("Age")
  pclass = tf.contrib.layers.real_valued_column("Pclass")
  siblings = tf.contrib.layers.real_valued_column("SibSp") # number of siblings (brother, sister, stepbrother, stepsister)
  parents = tf.contrib.layers.real_valued_column("Parch") # number of parents (Some children travelled only with a nanny, therefore parch=0 for them.)
  fare = tf.contrib.layers.real_valued_column("Fare") # passenger fare

  # bucketized base columns
  age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[5, 18, 25, 30, 35, 40, 45, 50, 55, 65])

  # crossed columns
  age_gender = tf.contrib.layers.crossed_column([age_buckets, gender], hash_bucket_size=int(1e4))

  columns = [name, gender, pclass, embarked, cabin, age, age_buckets, siblings, parents, fare, age_gender]

  # create the linear model
  model = tf.contrib.learn.LinearClassifier(model_dir=model_dir, feature_columns=columns)
  return model

def input_fn(df, is_training=False):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.

  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          # values=df[k].values,
          values=df[k].astype(str).values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}

  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  if is_training:
    # Converts the label column into a constant Tensor.
    # Returns the feature columns and the label.
    label = tf.constant(df[LABEL_COLUMN].values, shape=[df[LABEL_COLUMN].size, 1])
    return feature_cols, label
  else:
    return feature_cols

'''
  Print stats for the train and set data sets, then predict based on the test set (doesn't contain the “Survived” column)
'''
train, train_test = read_data_and_return_train_test(split=0.2) # 20% test, 80% train

# fit the model against the train set
model = build_estimator()
print("Training on train data:"+ str(train.shape))
model.fit(input_fn=lambda: input_fn(train, is_training=True), steps=600)

# evaluate model against the 20% of the train data “train_test”
results = model.evaluate(input_fn=lambda: input_fn(train_test, is_training=True), steps=1)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))

# use test data for the prediction
test = pd.read_csv("data/test.csv") # 418 people to predict
print("Test data shape: " + str(test.shape))

# show predictions for the test set (calculate the “Survived” column)
predictions = enumerate(model.predict(input_fn = lambda:input_fn(test, is_training=False)))
print("PassengerId,Survived")
for idx, prediction in predictions:
  print(str(test.iloc[idx].T["PassengerId"]) + "," + str(prediction))

# 0.76555 accuracy on kaggle
