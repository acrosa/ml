# coding: utf-8
from __future__ import print_function

import numpy as np
import tflearn

import pandas as pd

def read_csv(path):
  data = pd.read_csv(path)
  return data

def set_ignored(data):
  # remove columns we won't analyze
  return data.drop(['Name', 'Ticket'], axis=1)

def set_dummies(data):
  clone = data.copy()

  # embarked
  embarked = pd.get_dummies(clone.Embarked, prefix='embarked')
  clone = pd.concat([clone.drop(['Embarked'], axis=1), embarked], axis=1)

  # Pclass
  pclass = pd.get_dummies(clone.Pclass, prefix='pclass')
  clone = pd.concat([clone.drop(['Pclass'], axis=1), pclass], axis=1)

  # Sex
  sex = pd.get_dummies(clone.Sex, prefix='sex')
  clone = pd.concat([clone.drop(['Sex'], axis=1), sex], axis=1)

  # cabin
  clone['Cabin'] = clone['Cabin'].map(lambda c : c[0])
  cabin_dummies = pd.get_dummies(clone['Cabin'], prefix='cabin')
  clone = pd.concat([clone, cabin_dummies], axis=1)
  clone = clone.drop(['Cabin'], axis=1)

  clone['FamilySize'] = clone['Parch'] + clone['SibSp'] + 1  
  # introducing other features based on the family size
  clone['Singleton'] = clone['FamilySize'].map(lambda s : 1 if s == 1 else 0)
  clone['SmallFamily'] = clone['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
  clone['LargeFamily'] = clone['FamilySize'].map(lambda s : 1 if 5<=s else 0)

  return clone

def set_na(data):
  # some ages are NaN set the median age for those
  data['Age'].fillna(data['Age'].median(), inplace=True)
  data['Fare'].fillna(data['Fare'].mean(), inplace=True)
  data['Cabin'].fillna('U', inplace=True)
  return data

def process_data(data):
  data = set_ignored(data)
  data = set_na(data)
  data = set_dummies(data)
  return data

def read_combined_train_and_test():
  train = pd.read_csv("data/train.csv")
  test = pd.read_csv("data/test.csv")
  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop('index', inplace=True,axis=1)
  
  combined = process_data(combined)
  return combined.loc[0:train.shape[0]-1], combined.loc[train.shape[0]:]

'''
  Print stats for the train and set data sets, then predict based on the test set (doesn't contain the “Survived” column)
'''

train, test = read_combined_train_and_test()

# convert to format that tensorflow undestands
target = pd.get_dummies(train.Survived)
train = train.drop(['Survived'], axis=1)
data = np.array(train, dtype=np.float32)
target = np.array(target, dtype=np.float32)
total_features = data.shape[1]

# Build neural network
net = tflearn.input_data(shape=[None, total_features])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(data, target, n_epoch=100, batch_size=16, show_metric=True)
test = test.drop(['Survived'], axis=1)

test_data = np.array(test, dtype=np.float32)
test_data

results = model.predict(test_data)

# print csv format for submission
print("PassengerId,Survived")
i=0
for bad, good in results:
  survived = 1 if good > bad else 0
  print(str(int(test.iloc[i].T["PassengerId"])) + ","+ str(survived))
  i+=1

# 0.77990 accuracy on kaggle
