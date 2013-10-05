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
#import shelve

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
    #if x.shape[0] == 50000:# for all data points
    #calculate logp and the normalization constant Z
    #try:
    #if x.shape[0] == 50000 or x.shape[0] == 10000:
    #print np.dot(w.T,x.T).shape
    b = np.array([b,]*x.shape[0]).T
    #print b.shape
    #x shape 50000x784
    #print w.T.shape,x.T.shape
    logq = np.dot(w.T,x.T)  + b
    #print logq.shape
    Z = np.sum(np.exp(logq), axis=0)
    Z = np.array([Z,]*b.shape[0])
    #z = np.ones(b.shape[1])
    #z = np.log(Z) * z 
    logp = logq - Z
    #print logp
    #logp = np.exp(logp)
    #print Z.shape
    '''
      for i in range(x.shape[0]):
            for j in range(len(logq)):
                logq[j] = np.dot(w.T[j], x[i]) + b[j]
                Z += np.exp(logq[j])
            #calculate normalized probabilities
            for j in range(len(logp)):
                logp[j] = logq[j] - np.log(Z)
    '''
    '''
    else:#for one data point
        for j in range(len(logq)):
            logq[j] = np.dot(w.T[j], x) + b[j]
            Z += np.exp(logq[j])
        for j in range(len(logp)):
            logp[j] = logq[j] - np.log(Z)
    '''
    
    #print logp.shape
    return logp
        
def logreg_gradient(x, t, w, b):
    #init
    #print w.shape
    logq, logp, deltaq = np.zeros(b.shape[0]), np.zeros(b.shape[0]), np.zeros(b.shape[0])
    #print "a"    
    #print deltaq.shape    
    weights = np.zeros((x.shape[0], b.shape[0]))
    Z = 0 
    #calculate logp and the normalization constant Z
    #x shape 784
    logq = np.dot(w.T, x) + b
    '''
    for j in range(len(logq)):
        Z+= np.exp(logq[j])
    '''
    Z = np.sum(np.exp(logq))
    #logp = logq - np.log(Z)
    '''
    for j in range(len(logq)):
        logq[j] = np.dot(w.T[j], x) + b[j]
        Z += np.exp(logq[j])
    '''
    #calculate normalized probabilities
    '''    
    z = np.ones(b.shape[0])
    z = np.log(Z) * z 
    logp = logq - z
    '''
    '''
    for j in range(len(logp)):
        logp[j] = logq[j] - np.log(Z)
    '''
    #calculate delta
    deltaq[t] = 1 - ((1/Z)*np.exp(np.dot(w.T[t],x)+b[t]))
    #print deltaq[0:t]
    deltaq[0:t] = - ((1/Z)*np.exp(np.dot(w.T[0:t],x)+b[0:t]))
    #print deltaq[0:t]
    deltaq[t+1:len(deltaq)] = - ((1/Z)*np.exp(np.dot(w.T[t+1:len(deltaq)],x)+b[t+1:len(deltaq)]))
    '''
    test = deltaq
    for j in range(len(deltaq)):
        #if target class different delta
        if j == t:
            deltaq[j] = 1 - ((1/Z)*np.exp(np.dot(w.T[j],x)+b[j]))
        else:
            deltaq[j] = - ((1/Z)*np.exp(np.dot(w.T[j],x)+b[j]))
    '''
    #if (test==deltaq).all:
    #    print "yes"
    #calculate weight vector

    x = np.array([x])
    deltaq = np.array([deltaq])
    weights = np.dot(deltaq.T, x).T
    '''
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
           weights[i,j] = np.dot(deltaq[j], x[i])
    '''
    #calculate bias vector
    #bias = deltaq.reshape()
    bias = np.squeeze(np.asarray(deltaq))
    '''
    print "bias shape"
    print bias.shape
    '''
    #print weights.shape
    #print weights[1,1]
    return weights, bias#, logp
    
def sgd_iter(x_train, t_train, w, b):
    #shuffle indices
    index_shuf = range(len(x_train))
    shuffle(index_shuf)
    #set learning rate
    a = 10**(-4)
    #print a
    cnt = 0
    #cnt = -1
    #perform gradient ascent on all training data points
    #logp_train_bef = return_likelihood(x_train, t_train, w, b)
    for i in index_shuf:
        cnt += 1
        gradw, gradb = logreg_gradient(x_train[i], t_train[i], w, b)
        #print w[1,1]
        '''
        break1 = gradw[gradw>0.001]
        break2 = gradb[gradb>0.001]
        #break2 = [sublist for sublist in gradb if sublist[1] < 0.00001]
        #print break1
        if len(break1) == 0 & len(break2)== 0:
            print "breaking iteration, lower than threshold"
            break    print "x shape"
    print x.shape
        '''
        #print "not yet"
        #w_check = w + np.dot(a, gradw)
        #b_check = b + np.dot(a, gradb)
        w = w + a * gradw
        b = b + a * gradb
        #logp_train_aft = return_likelihood(x_train, t_train, w, b)
        '''
        if cnt == 1:
            logp_train_bef = logp
        elif np.allclose(logp, logp_train_bef, rtol = 0.1, atol = 0.001):
            #(np.abs(logp - logp_train_bef) < 0.1).all():
            print "threshold. breaking after processing "+str(cnt)+" datapoints..."
            break
        else:
            logp_train_bef = logp
        '''
        #break after cnt iterations
        #if cnt == 1000:
        #    break
    return w, b
    
def validate(x_valid, t_valid, w, b, numval):
    print "model built. validating..."
    #best, worst = np.zeros(8), np.zeros(8)
    validation = []
    #index = -1
    '''
    for i in range(len(x_valid)):
        logp = return_likelihood(x_valid[i],t_valid[i],w,b)
        validation.append((logp[t_valid[i]], i)) 
    '''
    logp = return_likelihood(x_valid,t_valid,w,b)
    #print logp.shape
    for t in range(len(t_valid)):
        print t_valid[t], t
        validation.append((logp[t_valid[t],t], t))
    #print validation[1]
    #validation.extend((logp[: , t_valid[:]], t_valid[:])) 
    #print validation[1]
    #sorts in ascending order so the last 8 have the greatest log likelihood
    validation = sorted(validation,key=itemgetter(0))
    print validation[len(validation)-numval:len(validation)]
    print 
    print logp[:,validation[len(validation)-1][1]]
    print logp[:,validation[0][1]]
    return validation[len(validation)-numval:len(validation)], validation[0:numval] 
    
def train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, epochs):
    #training, perform num_iter iterations 
    print "training..." 
    plt.figure("plot of conditional log-likelihood")
    logp_t = []
    logp_v = []
    #numrows = int(num_iter/2)
    for i in range(epochs):
        print "iteration: "+str(i+1)
        w, b = sgd_iter(x_train, t_train, w, b)
        logp_train = return_likelihood(x_train, t_train, w, b)
        logp_valid = return_likelihood(x_valid, t_valid, w, b)
        '''        
        #size of lop_train is (10,)
        logp_train = np.sum(logp_train, axis= 1)
        logp_valid = np.sum(logp_valid, axis= 1)
        '''
        #print logp_train.shape
        logp_t.append(np.mean(logp_train, dtype = np.float64))
        logp_v.append(np.mean(logp_valid, dtype = np.float64))
        #print logp_train.shape
        #print logp_valid.shape
        #print "before plot"
        #plt_log_like(logp_train, logp_valid, i, numrows)
        #print "out of plot"
    return w, b, logp_t, logp_v

def train_mlp(x_train, t_train, w, b):
    return "you wish"
    
def plt_log_like(logp_train, logp_valid ,num_iter, numrows):
    print "inside plot"
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

'''
#save training parameters, time consuming to train every time    
def save_params(params, location):
    filename = location
    my_shelf = shelve.open(filename,'n') # 'n' for new
    try:
        my_shelf['params']  = params 
    except TypeError:
        print('ERROR shelving')
    my_shelf.close()

def load_params(filename):
    params = {}
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        params = my_shelf[key]
        my_shelf.close()
    return params
'''
    
#get data
(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
#plot_digits(x_train[0:8], numcols=4)

#find number of classes and initialize weight and bias vectors
classes = np.unique(t_train)
w = np.zeros((x_train.shape[1], len(classes)))
b = np.zeros(len(classes))

'''
decide if you want to train the model again 
or choose the previous trained model
'''
#train = 1
#params = {}
#location = '/home/pathos/UvA/MachineLearning/Homework/Lab3/shelved_params.data'
'''
if train == 1:
    w, b, logp_t, logp_v = train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, 10)
    plt.plot(logp_t, color = 'b', label = 'training')
    plt.plot(logp_v, color = 'g', label = 'validation')  
    plt.legend()
    params['w'] = w
    params['b'] = b
    save_params(params, location)
else:
    params = load_params(location)
    w = params['w']
    b = params['b']
'''
w, b, logp_t, logp_v = train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, 5)
plt.plot(logp_t, color = 'b', label = 'training')
plt.plot(logp_v, color = 'g', label = 'validation') 
#plt.ylim([0,1]) 
plt.legend()
#print w.T[1]

#plot weights  
plt.figure("weights")  
plot_digits(w.T, numcols=5)


#validate and plot images with the best and worst 8 probabilities
best, worst = validate(x_valid, t_valid, w, b, 16)
#print best
plot_res_reg(best, worst)

#units in hidden layer for mlp
L = 10
classes = np.unique(t_train)
v = np.zeros((x_train.shape[1], L))
a = np.zeros(L)

plt.show()

