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
    #print "plot"
    #print data.shape
    #data = data.T
    numdigits = data.shape[0]
    #print numdigits
    numrows = int(numdigits/numcols)
    #plt.imshow(data.reshape(shape), interpolation = 'nearest', cmap = 'Greys')
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    #plt.show()
    
def return_likelihood(x, t, w, b):
    logq, logp = np.zeros(b.shape[0]), np.zeros(b.shape[0])
    Z = 0 
    try:# for all data points
        #calculate logp and the normalization constant Z
        for i in range(x.shape[0]):
            for j in range(len(logq)):
                logq[j] = np.dot(w.T[j], x[i]) + b[j]
                Z += np.exp(logq[j])
            #calculate normalized probabilities
            for j in range(len(logp)):
                logp[j] = logq[j] - np.log(Z)
    except:#for one data point
        for j in range(len(logq)):
            logq[j] = np.dot(w.T[j], x) + b[j]
            Z += np.exp(logq[j])
        for j in range(len(logp)):
            logp[j] = logq[j] - np.log(Z)
    return logp
        
def logreg_gradient(x, t, w, b):
    #init
    logq, logp, deltaq = np.zeros(b.shape[0]), np.zeros(b.shape[0]), np.zeros(b.shape[0])
    weights = np.zeros((x.shape[0], b.shape[0]))
    Z = 0 
    #calculate logp and the normalization constant Z
    for j in range(len(logq)):
        logq[j] = np.dot(w.T[j], x) + b[j]
        Z += np.exp(logq[j])
    #calculate normalized probabilities
    for j in range(len(logp)):
        logp[j] = logq[j] - np.log(Z)
    #calculate delta
    for j in range(len(deltaq)):
        #if target class different delta
        if j == t:
            deltaq[j] = 1 - ((1/Z)*np.exp(np.dot(w.T[j],x)+b[j]))
        else:
            deltaq[j] = - ((1/Z)*np.exp(np.dot(w.T[j],x)+b[j]))
    #calculate weight vector
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
           weights[i,j] = np.dot(deltaq[j], x[i])
    #calculate bias vector
    bias = deltaq
    #print weights[1,1]
    return weights, bias
    
def sgd_iter(x_train, t_train, w, b):
    #shuffle indices
    index_shuf = range(len(x_train))
    shuffle(index_shuf)
    #set learning rate
    a = 1*np.exp(-4)
    cnt = -1
    #perform gradient ascent on all training data points
    for i in index_shuf:
        cnt += 1
        gradw, gradb = logreg_gradient(x_train[i], t_train[i], w, b)
        #print w[1,1]
        w = w + np.dot(a, gradw)
        b = b + np.dot(a, gradb)
        #break after cnt iterations
        if cnt == 100:
            break
    return w, b
    
def validate(x_valid, t_valid, w, b, numval):
    print "model built. validating..."
    #best, worst = np.zeros(8), np.zeros(8)
    validation = []
    #index = -1
    for i in range(len(x_valid)):
        logp = return_likelihood(x_valid[i],t_valid[i],w,b)
        validation.append((logp[t_valid[i]], i)) 
    validation = sorted(validation,key=itemgetter(0))
    #print validation
    return validation[len(validation)-numval:len(validation)], validation[0:numval] 
    
def train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, num_iter):
    #training, perform num_iter iterations 
    print "training..." 
    plt.figure("plot of conditional log-likelihood")
    numrows = int(num_iter/2)
    for i in range(num_iter):
        print "iteration: "+str(i+1)
        w, b = sgd_iter(x_train, t_train, w, b)
        logp_train = return_likelihood(x_train, t_train, w, b)
        logp_valid = return_likelihood(x_valid, t_valid, w, b)
        plt_log_like(logp_train, logp_valid, i, numrows)
    return w, b

def train_mlp(x_train, t_train, w, b):
    return "you wish"
    
def plt_log_like(logp_train, logp_valid, num_iter, numrows):
    plt.subplot(numrows+1, 2, num_iter+1)
    plt.plot(logp_train, color = 'b', label = "training, "+str(num_iter+1))
    plt.plot(logp_valid, color = 'g', label = "validation, "+str(num_iter+1))
    plt.legend(loc = 4, prop={'size':6})
    
def plot_res_reg(best, worst):    
    #ind_best = [x[1] for x in best]
    #ind_worst = [x[1] for x in worst]
    bestshow = np.array([x_valid[x[1]] for x in best])
    worstshow = np.array([x_valid[x[1]] for x in worst])
    
    plt.figure("best")
    plot_digits(bestshow, numcols = 4)
    plt.figure("worst")
    plot_digits(worstshow, numcols = 4)

def sigmoid(a):
     return 1 / (1 + np.exp(-a))
    
def calc_hidden(x, v, a):
    h = np.zeros(v.shape[0])
    for j in range(len(v)):
        h[j] = sigmoid(np.dot(v.T[j], x) + a[j])
    return h
    
#get data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
#plot_digits(x_train[0:8], numcols=4)

#find number of classes and initialize weight and bias vectors
classes = np.unique(t_train)
w = np.zeros((x_train.shape[1], len(classes)))
b = np.zeros(len(classes))

w, b = train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, 5)

#print w.T[1]

#plot weights  
plt.figure("weights")  
plot_digits(w.T, numcols=5)


#validate and plot images with the best and worst 8 probabilities
best, worst = validate(x_valid, t_valid, w, b, 8)
#print best
plot_res_reg(best, worst)

#units in hidden layer for mlp
L = 10
classes = np.unique(t_train)
v = np.zeros((x_train.shape[1], L))
a = np.zeros(L)



plt.show()

