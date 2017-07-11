#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tempfile

from six.moves import urllib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
  combined.drop('index',inplace=True,axis=1)
  
  combined = process_data(combined)
  return combined.loc[0:890], combined.loc[891:]

'''
  Print stats for the train and set data sets, then predict based on the test set (doesn't contain the “Survived” column)
'''
train, test = read_combined_train_and_test()

# set target values for training
targets = train.Survived

# now drop them from the training set
train.drop('Survived', axis=1, inplace=True)
test.drop('Survived', axis=1, inplace=True)

# fit the model against the train set
print("Training on train data:"+ str(train.shape))
forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }
grid_search = GridSearchCV(forest, parameter_grid, cv=None) # note cv is not a great cross validation strategy

# fit time!
grid_search.fit(train, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# use test data for the prediction
print(test.head())

print("Test data shape: " + str(test.shape))
output = grid_search.predict(test).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output

print("PassengerId,Survived")
for index, row in df_output.iterrows():
  print(str(row['PassengerId']) + "," + str(row['Survived']))

# 0.78469 accuracy on kaggle
