# This file contains a series of functions that will be important
# in running logistic regression.
#
# The functions are made under the assumption that array data types
# are being used for variables: X, Y, theta
#
# Info on expected parameters for the data:
#
#   X: Array, dimensions (m,n) where m is the number of data points
#      and n is the number of features per point.
#
#   Y: Array, dimensions (m,1). (Expected output for each datapoint)
#
#   theta: Array, dimensions (1,n). (Feature Vector)
#
#   alpha: Integer. Controls the size of a step of gradient descent.
#
#   lambda: Integer. Controls regularization. (Not implimented yet)


###################################################################
#Modules
from math import e
from math import log
import numpy
from scipy.optimize import fmin_cg
from random import randint

###################################################################
#Basic functions

#Sigmoid Function
def sig(X):
    
    return(1/(1+(e**(-X))))


#m-val Function (fairly costly, so best to calculate outside of
# cost function and then resubmit it into the cost function so
# that it only has to be run once.)
def mval(X):
    
    return((numpy.shape(X))[0])


#n-val Function
def nval(X):

    return((numpy.shape(X))[1])



###################################################################
#Functions

#Cost Function
def CostFunction(theta,*args):

    X,Y,alpha,m = args
    h_of_x = sig((X.dot(theta))/10000)
    
    Cost = (1/m)*sum( ((-Y)*numpy.log(h_of_x)) - ((1-Y)*numpy.log(1-h_of_x)) )

    return(Cost)

#Gradient of Cost Function
def CostPrime(theta,*args):

    X,Y,alpha,m = args
    h_of_x = sig((X.dot(theta))/10000)

    Gradient = theta - (alpha*((X.T).dot(h_of_x-Y)))

    return(Gradient)


###################################################################
#Data


#alpha
alpha = 10

#X
X = numpy.genfromtxt("C:\\Users\\Sean\\Desktop\\Kaggle\\Dogs vs Cats Redux\\Algorithms\\Logistic Regression\\Trial 2 5000x2500\\5000x2500 Data.txt",delimiter=',')

#m
m = mval(X)

#nval
n = nval(X)

#Y
Y = numpy.array([X[i][n-1] for i in range(m)])

#theta
theta = numpy.array([randint(0,100)/100 for w in range(n)])


###################################################################
#Execution


#Here is an implimentation of the minimization funciton

args = (X,Y,alpha,m)

Result = fmin_cg(CostFunction,theta,fprime=CostPrime,args=args)

print(CostFunction(theta, X, Y, alpha, m))
print(CostPrime(theta, X, Y, alpha, m))


