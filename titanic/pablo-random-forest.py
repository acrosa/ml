'''
  Ejemplo de random forest hecho por @fernandezpablo (en 5 min. en su cama) 
'''
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test_orig = pd.read_csv('test.csv')

def drop_useless_colums(df):
    return df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

def add_dummies(df):
    clone = df.copy()
    genders = pd.get_dummies(clone.Sex, prefix='is')
    clone = pd.concat([clone.drop(['Sex'], axis=1), genders], axis=1)

    embarked = pd.get_dummies(clone.Embarked, prefix='embarked')
    return pd.concat([clone.drop(['Embarked'], axis=1), embarked], axis=1)

def handle_nas(df):
    df.Age = df.Age.fillna(-1)
    df.Fare = df.Fare.fillna(0)
    return df

train = drop_useless_colums(train)
train = add_dummies(train)
train = handle_nas(train)

y = train.Survived
X = train.drop(['Survived'], axis=1)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
clf = ExtraTreesClassifier(n_estimators=250, max_depth=5)
clf.fit(Xtrain, ytrain, sample_weight=np.where(ytrain, 1 + ytrain.mean(), 1))
print clf.score(Xtrain, ytrain)
print clf.score(Xtest, ytest)

clf.fit(X, y, sample_weight=np.where(y, 1 + y.mean(), 1))
test = drop_useless_colums(test_orig)
test = add_dummies(test)
test = handle_nas(test)

ypred = clf.predict(test)