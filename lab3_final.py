# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:07:31 2013

@author: pathos
"""
from __future__ import division
from random import shuffle
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import gzip, cPickle 

def load_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	data = cPickle.load(f)
	f.close()
	return data

def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    #plt.show()
    
def return_likelihood(x, t, w, b):
   #return the normalized log probability of all datapoints in a dataset
   logq, logp = np.zeros(b.shape[0]), np.zeros(b.shape[0])
   logp_true = np.zeros(t.shape[0])
   b = np.array([b,]*x.shape[0]).T
   logq = np.dot(w.T,x.T)  + b
   Z = np.sum(np.exp(logq), axis=0)
   Z = np.array([Z,]*b.shape[0])
   logp = logq - np.log(Z)
   #return only for true class labels
   for i in range(len(t)):
       logp_true[i] = logp[t[i],i] 
   return logp_true
        
def logreg_gradient(x, t, w, b):
    logq, logp, deltaq = np.zeros(b.shape[0]), np.zeros(b.shape[0]), np.zeros(b.shape[0]) 
    gradw = np.zeros((x.shape[0], b.shape[0]))
    
    logq = np.dot(w.T, x) + b
    Z = np.sum(np.exp(logq))

    #calculate delta
    deltaq[t] = 1 - ((1/Z)*np.exp(np.dot(w.T[t],x)+b[t]))
    deltaq[0:t] = - ((1/Z)*np.exp(np.dot(w.T[0:t],x)+b[0:t]))
    deltaq[t+1:len(deltaq)] = - ((1/Z)*np.exp(np.dot(w.T[t+1:len(deltaq)],x)+b[t+1:len(deltaq)]))
    
    #find gradient of the weights and bias, according to a datapoint
    x = np.array([x])
    deltaq = np.array([deltaq])
    gradw = np.dot(deltaq.T, x).T
    gradb = np.reshape(deltaq,-1)
    
    return gradw, gradb
    
def sgd_iter(x_train, t_train, w, b):
    #shuffle indices
    index_shuf = range(len(x_train))
    shuffle(index_shuf)
    #set learning rate
    a = 10**(-4)
    
    #perform one iteration through the training set and 
    #update the parameters according to gradient ascent
    for i in index_shuf:
        gradw, gradb = logreg_gradient(x_train[i], t_train[i], w, b)
        w = w + a * gradw
        b = b + a * gradb
        
    return w, b
    
def validate(x_valid, t_valid, w, b, numval):
    print "model built. validating..."
    validation = []
    logp = return_likelihood(x_valid,t_valid,w,b)
    for t in range(len(t_valid)):
        validation.append((logp[t], t))
        
    #sorts in ascending order so the last 8 have the greatest log likelihood
    validation = sorted(validation,key=itemgetter(0))
    #return the numval best and worst digits
    return validation[len(validation)-numval:len(validation)], validation[0:numval] 
    
def train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, epochs):
    #training, perform iterations equal to epochs
    print "training..." 
    plt.figure("plot of conditional log-probability")
    logp_t = []
    logp_v = []

    for i in range(epochs):
        print "iteration: "+str(i+1)
        w, b = sgd_iter(x_train, t_train, w, b)
        
        #get the log probabilities after each iteration
        logp_train = return_likelihood(x_train, t_train, w, b)
        logp_valid = return_likelihood(x_valid, t_valid, w, b)
        
        #append the average of them
        logp_t.append(np.mean(logp_train))
        logp_v.append(np.mean(logp_valid))
    
    return w, b, logp_t, logp_v

    
def plot_res_reg(best, worst):
    #plot the best and worst digits
    bestshow = np.array([x_valid[x[1]] for x in best])
    worstshow = np.array([x_valid[x[1]] for x in worst])
    
    plt.figure("best")
    plot_digits(bestshow, numcols = 4)
    plt.figure("worst")
    plot_digits(worstshow, numcols = 4)


    
#get data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
#plot_digits(x_train[0:8], numcols=4)

#find number of classes and initialize weight and bias vectors
classes = np.unique(t_train)
w = np.zeros((x_train.shape[1], len(classes)))
b = np.zeros(len(classes))


#uncomment below for multiclass logistic regression
'''
w, b, logp_t, logp_v = train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, 2)
plt.plot(logp_t, color = 'b', label = 'training')
plt.plot(logp_v, color = 'g', label = 'validation') 
plt.legend()

#plot weights  
plt.figure("weights")  
plot_digits(w.T, numcols=5)


#validate and plot digits with the best and worst 8 probabilities
best, worst = validate(x_valid, t_valid, w, b, 8)
plot_res_reg(best, worst)
'''

#units in hidden layer for mlp
L = 10
# For the sigmoid function this interval initialization ensures that, early in training, 
#each neuron operates in a regime of its activation function where information can easily 
#be propagated both upward (activations flowing from inputs to outputs) and backward 
#(gradients flowing from outputs to inputs). from : http://deeplearning.net/tutorial/mlp.html
v = np.asarray(np.random.uniform(-np.sqrt(6. / (x_train.shape[1] + L)),np.sqrt(6. / (x_train.shape[1] + L)),(x_train.shape[1], L)))
#v = np.random.randn(x.shape[1], L)
a = np.zeros(L)
w = np.zeros((L, len(classes)))
b = np.zeros(len(classes))


def train_mlp(x_train, t_train, x_valid, t_valid, w, b, v, a, epochs):
    #training, perform iterations equal to epochs
    print "training mlp ..." 
    plt.figure("plot of conditional log-probability")
    logp_t = []
    logp_v = []

    for i in range(epochs):
        print "iteration: "+str(i+1)
        w, b, v, a = sgd_iter_mlp(x_train, t_train, w, b, v, a)
        #uncomment below for using validation set as well for training
        #w, b, v, a = sgd_iter_mlp(x_valid, t_valid, w, b, v, a)
        logp_train = return_likelihood_mlp(x_train, t_train, w, b, v, a)
        logp_valid = return_likelihood_mlp(x_valid, t_valid, w, b, v, a)
        
        logp_t.append(np.mean(logp_train))
        logp_v.append(np.mean(logp_valid))

    return w, b, v, a, logp_t, logp_v

def sgd_iter_mlp(x_train, t_train, w, b, v, a):
    index_shuf = range(len(x_train))
    shuffle(index_shuf)
    #set learning rate
    lr = 10**(-4)
    
    for i in index_shuf:
        gradw, gradb, gradv, grada = gradient_mlp(x_train[i], t_train[i], w, b, v, a)
        
        w = w + lr * gradw
        b = b + lr * gradb
        v = v + lr * gradv
        a = a + lr * grada
        
    return w, b, v, a
    
def sigmoid(a):
     return 1 / (1 + np.exp(-a))
    
def calc_hidden(x, v, a):
    h = sigmoid(np.dot(v.T, x) + a)
    return h


def gradient_mlp(x, t, w, b, v, a):
    #init
    logq, logp, deltaq = np.zeros(b.shape[0]), np.zeros(b.shape[0]), np.zeros(b.shape[0])
    deltah = np.zeros(v.shape[1])     
    gradw = np.zeros((x.shape[0], b.shape[0]))
    gradv = np.zeros(v.shape)
    grada = np.zeros(a.shape[0])
    #calculate logp and the normalization constant Z
    h = calc_hidden(x, v, a).T

    
    logq = np.dot(w.T, h) + b

    Z = np.sum(np.exp(logq))
    
    deltaq[t] = 1 - ((1/Z)*np.exp(np.dot(w.T[t],h)+b[t]))
    deltaq[0:t] = - ((1/Z)*np.exp(np.dot(w.T[0:t],h)+b[0:t])) 
    deltaq[t+1:len(deltaq)] = - ((1/Z)*np.exp(np.dot(w.T[t+1:len(deltaq)],h)+b[t+1:len(deltaq)]))  
    deltaq = np.array([deltaq]) 
    
    h = np.array([h]).T
    deltah = np.dot(deltaq,w.T)
    x = np.array([x])
    
    #gradients for the weights and biases for the output layer
    gradw = np.dot(h,deltaq)
    gradb = np.reshape(deltaq,-1)

    #gradient for the weights v of the hidden layer
    I = np.matrix(np.identity(v.shape[1]))
    diagonal = np.dot(np.dot(np.dot(deltah.T, h.T),I),h)
    diagonal = np.reshape(np.asarray(diagonal), -1)
    gradv = np.dot(v, np.diag(diagonal))  
    
    #gradient for the biases a for the hidden layer    
    grada = np.dot(np.dot(deltah.T,h.T),1-h)
    grada = np.reshape(grada,-1)
    
    return gradw, gradb, gradv, grada
    
def return_likelihood_mlp(x, t, w, b, v, a):
   #return conditional log probability of all datapoints in the dataset
   logq, logp = np.zeros(b.shape[0]), np.zeros(b.shape[0])  
   logp_true = np.zeros(t.shape[0])
   b = np.array([b,]*x.shape[0]).T
   a = np.array([a,]*x.shape[0]).T
   h = calc_hidden(x.T, v, a)
   logq = np.dot(w.T, h) + b
    
   Z = np.sum(np.exp(logq), axis=0)
   Z = np.array([Z,]*b.shape[0])
   
   logp = logq - np.log(Z)
   
   for i in range(len(t)):
       logp_true[i] = logp[t[i],i] 
   return logp_true
   
def validate_mlp(x_valid, t_valid, w, b, v, a, numval):
    print "model built. validating..."
    validation = []
    logp = return_likelihood_mlp(x_valid,t_valid,w,b,v,a)
    for t in range(len(t_valid)):
        validation.append((logp[t], t))

    validation = sorted(validation,key=itemgetter(0))

    return validation[len(validation)-numval:len(validation)], validation[0:numval] 

#uncomment for multilayer perceptron

w, b, v, a, logp_t, logp_v = train_mlp(x_train, t_train, x_valid, t_valid, w, b, v, a, 4)
plt.plot(logp_t, color = 'b', label = 'training')
plt.plot(logp_v, color = 'g', label = 'validation') 
plt.legend()

#print the final weights
#plt.figure("weights")  
#plot_digits(w.T, numcols=5)


#validate and plot images with the best and worst 8 probabilities
best, worst = validate_mlp(x_valid, t_valid, w, b, v, a, 8)
plot_res_reg(best, worst)

plt.show()

