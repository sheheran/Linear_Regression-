#Import libraries
import pandas as pd
import numpy as np
import math

#reading data files
#import train set
trainSet = 'IRIS_trainset.csv'
trainNames = ['sepal_lenght','sepal_width','petal_lenght','pedal_lenght','species']
train_data = pd.read_csv(trainSet, names=trainNames)
print(train_data.head(2))
#import test set
testSet = 'IRIS_testset.csv'
testNames = ['sepal_lenght','sepal_width','petal_lenght','pedal_lenght','species']
test_data = pd.read_csv(testSet, names=testNames)
print(test_data.head(2))

#combine train and test data
train_test_data = [train_data,test_data]
#mapping the labels
mapping_lbl = {'setosa':0.2, 'versicolor':0.4, 'virginica':0.6}
for dataset in  train_test_data:
	dataset['species'] = dataset['species'].map(mapping_lbl)

#splitting features and labels
X_train = train_data.drop('species', axis=1)
X_test = test_data.drop('species', axis=1)

#adding colum of 1s
m,n = X_train.shape
x_0 = np.ones((m,1))
X_train = np.hstack((x_0,X_train))
x_train = [X_train]
#adding colum of 1s
m,n = X_test.shape
x_0 = np.ones((m,1))
X_test = np.hstack((x_0,X_test))
x_test = [X_test]

Y_train = train_data['species']
y_train = [Y_train]
Y_test = test_data['species']
y_train = [Y_train]

def cal_cost(theta,X_train,Y_train):
	m = len(Y_train)
	predic = np.dot(X_train,theta)
	cost = (1/2*m) * np.sum(np.square(predic - y_train))
	return cost

theta = np.random.randn(5,1)
print(cal_cost(theta,X_train,Y_train))