#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
#train test split
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

#import csv files using pandas
data = 'IRIS.csv'
dataNames = ['sepal_lenght','sepal_width','petal_lenght','pedal_width','species']
Data = pd.read_csv(data, names=dataNames)
#print(Data.shape)
#print(Data.head(10))

#visualize relationship between features and response using scatterplots
sns.pairplot(Data, x_vars=['sepal_lenght','sepal_width','petal_lenght','pedal_width'], y_vars = 'species', size=7, aspect=0.7)
plt.show()
train_test_data = [Data]
#mapping labels
mapping_lbl = {'setosa':0.2, 'versicolor':0.4, 'virginica':0.6}
for dataset in train_test_data:
	dataset['species'] = dataset['species'].map(mapping_lbl)

#split the table for features and labels
train_data = Data.drop('species' ,axis=1)
target = Data['species']

#considering linear regression
#adding a colum of ones to the feature set
m,n = train_data.shape
ones = np.ones((m,1))
train_data = np.hstack((ones, train_data))
#train_data is a numpy array

#split train test data sets
X_train,X_test,Y_train,Y_test = train_test_split(train_data,target,test_size=0.2)

y_train = np.array([Y_train], dtype=np.float64)
y_train = y_train.T
#cost function
def cal_cost(theta,x_train,y_train):
	m = len(y_train)
	prediction = np.dot(x_train,theta)
	cost = (1 /2*m) * np.sum(np.square(prediction - y_train))
	return cost

theta = np.random.randn(5,1)
cal_cost(theta,X_train,y_train)


#parameter calculation
def gradient_descent(x, y, theta, iterations=100,learning_rate=0.01):
	#x - feature matrix
	#y - labels
	# iteration - running turms
	m = len(y) #number of data
	cost_history = np.zeros(iterations)
	theta_history = np.zeros((iterations,5))
	for i in range(iterations):

		#predicted values
		predict = np.dot(x,theta)
		theta = theta - (1/m)*learning_rate*(x.T.dot((predict - y)))
		theta_history[i,:] = theta.T
		cost_history[i] = cal_cost(theta,x,y)
		
	return theta, cost_history, theta_history

tr = 0.01
n_itr = 100
theta,cost_history,theta_history = gradient_descent(X_train,y_train,theta,n_itr,tr)
print(theta)
X = cost_history
Y = list(range(1,(n_itr+1)))
figsize = (10, 8)
plt.plot(X, Y, 'o', ls='-')
plt.show()

#maual hypothesis calculation
