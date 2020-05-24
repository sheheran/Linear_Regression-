# Linear_Regression-
The algorithm of linear regression &amp; python code
Applying Linear Regression
Considering a data set
 
X	Y
1	4
2	5
3	6
4	7
5	8
6	9
	When considering a Linear Regression problem, the hypothesis is represented as

y=θ_1 x+θ_0



On the left side of the table x values are named as features while the y values are called labels. There-fore above problem has one feature and a label. The hypothesis represents the line that these data point exist.

The purpose of linear regression is to find the best fit line to these data set, and when a feature (x) value is given which have no label defined using the predicted ML model it is possible to find the unknown label.
 //The code for setting up the data set
When finding the best-fit line we first assume some values for θ_1 and θ_0. When coding we assume the hypothesis to be in the form of 
y=θ_1 x_1+θ_0 x_0
Here feature represented as x_0 is equal to 1, there-for there will be no change in the hypothesis function. The new data set will be represented as 
X0	X	Y
1	1	4
1	2	5
1	3	6
1	4	7
1	5	8
1	6	9

// code for adding a row of ones 
Calculating hypothesis prediction with assumed x_1 & x_0. It’s represented as a multiplication of matrix. 
(■(x_0&x_1@x_0^'&x_1^' ))  .(█(θ_0@θ_1 ))= (█(y_1@y_2 ))
//code for predicting y0






Squired error
1/2m ∑_(i=1)^m▒〖(h(x)-y)〗^2 
The mean is halved (1/2) as a convenience for the computation of gradient descent as the derivative terms of the squired function will cancel out.
The squired error calculates the error between the predicted label and actual label.
//code for the cost function
Gradient descent
For finding correct θ_1 and θ_0 values, the gradient descent process is used.
  θ_0=  δ/m ∑_(i=1)^m▒〖(h(x)-y).  x_0 〗
θ_1=  δ/m ∑_(i=1)^m▒〖(h(x)-y)〗.  x_1  
The gradient of decent is represented as multiplication of matrix.

(█(θ_1@θ_2 ))=(█(θ_1@θ_2 ))- (δ/m)  .(■(x_0&x_1@x_0^'&x_1^' )).(h(x)-y)


 
 



