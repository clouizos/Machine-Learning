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
   
   #logq, logp, deltaq = np.zeros(b.shape[0]), np.zeros(b.shape[0]), np.zeros(b.shape[0])
   #print "a"    
   #print deltaq.shape    
   #gradw = np.zeros((x.shape[0], b.shape[0]))
   #Z = 0 
   #calculate logp and the normalization constant Z
   #x shape 784
   '''
   for i in range(len(x)):
       logq = np.dot(w.T, x[i]) + b
       Z = np.sum(np.exp(logq))
       logp += logq - np.log(Z)
   '''
   
   logq, logp = np.zeros(b.shape[0]), np.zeros(b.shape[0])
   #Z = 0 
   #if x.shape[0] == 50000:# for all data points
   #calculate logp and the normalization constant Z
   #try:
   #if x.shape[0] == 50000 or x.shape[0] == 10000:
   #print np.dot(w.T,x.T).shape
   b = np.array([b,]*x.shape[0]).T
   #print b
   #print b.shape
   #x shape 50000x784
   #print w.T.shape,x.T.shape
   logq = np.dot(w.T,x.T)  + b
   Z = np.sum(np.exp(logq), axis=0)
   Z = np.array([Z,]*b.shape[0])
   #z = np.ones(b.shape[1])
   #z = np.log(Z) * z 
   logp = logq - np.log(Z)
   #print logp
   
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
    gradw = np.zeros((x.shape[0], b.shape[0]))
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
#    print deltaq   
#    print
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
    #print x.shape
    #print deltaq.shape
    x = np.array([x])
    deltaq = np.array([deltaq])
    #print x.shape
    #print deltaq.shape
    #print deltaq
    gradw = np.dot(deltaq.T, x).T
    #print np.unique(gradw)
    '''
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
           weights[i,j] = np.dot(deltaq[j], x[i])
    '''
    #calculate bias vector
    #bias = deltaq.reshape()
    gradb = np.reshape(deltaq,-1)
    #gradb = np.squeeze(np.asarray(deltaq))
    '''
    print "bias shape"
    print bias.shape
    '''
    #print weights.shape
    #print weights[1,1]
    return gradw, gradb#, logp
    
def sgd_iter(x_train, t_train, w, b):
    #shuffle indices
    index_shuf = range(len(x_train))
    shuffle(index_shuf)
    #set learning rate
    a = 10**(-4)
    #print a
    #cnt = 0
    #cnt = -1
    #perform gradient ascent on all training data points
    #logp_train_bef = return_likelihood(x_train, t_train, w, b)
    for i in index_shuf:
        #cnt += 1
        gradw, gradb = logreg_gradient(x_train[i], t_train[i], w, b)
        #print w[1,1]
        '''
        break1 = gradw[gradw>0.001]
        break2 = gradb[gradb>0.001]
        #break2 = [sublist for sublist in gradb if sublist[1] < 0.00001]
        #print break1
        if len(break1) == 0 & len(break2)== 0:
            print "breaking iteration, lower than threshold"
            break    
        print "x shape"
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
        #print t_valid[t], t
        validation.append((logp[t_valid[t],t], t))
    #print validation[1]
    #validation.extend((logp[: , t_valid[:]], t_valid[:])) 
    #print validation[1]
    #sorts in ascending order so the last 8 have the greatest log likelihood
    validation = sorted(validation,key=itemgetter(0))
    #print validation[len(validation)-numval:len(validation)]
#    print 
#    print logp[:,validation[len(validation)-1][1]]
#    print logp[:,validation[0][1]]
#    print validation
    return validation[len(validation)-numval:len(validation)], validation[0:numval] 
    
def train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, epochs):
    #training, perform num_iter iterations 
    print "training..." 
    plt.figure("plot of conditional log-probability")
    logp_t = []
    logp_v = []
    #init values
#    logp_train = return_likelihood(x_train, t_train, w, b)
#    logp_valid = return_likelihood(x_valid, t_valid, w, b)
#    logp_t.append(np.mean(logp_train, dtype = np.float64))
#    logp_v.append(np.mean(logp_valid, dtype = np.float64))
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
#        print logp_train.shape
#        train = 0
#        valid = 0
#        for i in range(logp_train.shape[1]):
#            train += np.sum(logp_train[:,i])
#        train = train * 1/logp_train.shape[1]
#        logp_t.append(train)
        logp_t.append(np.mean(logp_train))
#        for i in range(logp_valid.shape[1]):
#            valid += np.sum(logp_valid[:,i])
#        valid = valid * 1/logp_valid.shape[1]
#        logp_v.append(valid)
        logp_v.append(np.mean(logp_valid))
        #print logp_train.shape
        #print logp_valid.shape
        #print "before plot"
        #plt_log_like(logp_train, logp_valid, i, numrows)
        #print "out of plot"
    return w, b, logp_t, logp_v

    
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
#uncomment below for multiclass logistic regression
'''
w, b, logp_t, logp_v = train_mult_log_reg(x_train, t_train, x_valid, t_valid, w, b, 30)
plt.plot(logp_t, color = 'b', label = 'training')
plt.plot(logp_v, color = 'g', label = 'validation') 
#plt.ylim([0,1]) 
plt.legend()
#print w.T[1]

#plot weights  
plt.figure("weights")  
plot_digits(w.T, numcols=5)


#validate and plot images with the best and worst 8 probabilities
best, worst = validate(x_valid, t_valid, w, b, 8)
#print best
plot_res_reg(best, worst)
'''

#units in hidden layer for mlp
L = 10
v = np.asarray(np.random.uniform(-np.sqrt(6. / (x_train.shape[1] + L)),np.sqrt(6. / (x_train.shape[1] + L)),(x_train.shape[1], L)))
#v = np.random.randn(x.shape[1], L)
a = np.zeros(L)
w = np.zeros((L, len(classes)))
b = np.zeros(len(classes))

def train_mlp(x_train, t_train, w, b, v, a, epochs):
     #training, perform num_iter iterations 
    print "training mlp ..." 
    plt.figure("plot of conditional log-probability")
    logp_t = []
    logp_v = []

    for i in range(epochs):
        print "iteration: "+str(i+1)
        w, b, v, a = sgd_iter_mlp(x_train, t_train, w, b, v, a)
        print "one full gradient descent"
        logp_train = return_likelihood_mlp(x_train, t_train, w, b, v, a)
        logp_valid = return_likelihood_mlp(x_valid, t_valid, w, b, v, a)
        print "found log likelihoods"
        logp_t.append(np.mean(logp_train))

        logp_v.append(np.mean(logp_valid))

    return w, b, v, a, logp_t, logp_v

def sgd_iter_mlp(x_train, t_train, w, b, v, a):
    index_shuf = range(len(x_train))
    shuffle(index_shuf)
    #set learning rate
    lr = 10**(-4)
    
    for i in index_shuf:
        gradw, gradb, gradv, grada = logreg_gradient_mlp(x_train[i], t_train[i], w, b, v, a)
        
        w = w + lr * gradw
        b = b + lr * gradb
        v = v + lr * gradv
        #print "sgd"
        #print a.shape, grada.shape
        a = a + lr * grada
        
    return w, b, v, a
    
def sigmoid(a):
     return 1 / (1 + np.exp(-a))
    
def calc_hidden(x, v, a):
    #h = np.zeros(v.shape[1])
#    print "hidden"
#    print x.shape, v.shape, a.shape
    h = sigmoid(np.dot(v.T, x) + a)
#    print "ha"
#    print h.shape
    '''
    for j in range(h.shape[0]):
        h[j] = sigmoid(np.dot(v.T[j], x) + a[j])
    print h.shape
    '''
    return h
    #return np.asarray([h]).T

def logreg_gradient_mlp(x, t, w, b, v, a):
#    print "once"
    #init
    logq, logp, deltaq = np.zeros(b.shape[0]), np.zeros(b.shape[0]), np.zeros(b.shape[0])
    deltah = np.zeros(v.shape[1])     
    gradw = np.zeros((x.shape[0], b.shape[0]))
    gradv = np.zeros(v.shape)
#    print v.shape
    grada = np.zeros(a.shape[0])
    #Z = 0 
    #calculate logp and the normalization constant Z
    #x shape 784
    h = calc_hidden(x, v, a).T
#    print "aaaaa"
#    print h.shape
    
    logq = np.dot(w.T, h) + b
#    print logq.shape
#    print 'bbb'
    Z = np.sum(np.exp(logq))
    #print deltaq.shape
    #calculate deltas
    deltaq[t] = 1 - ((1/Z)*np.exp(np.dot(w.T[t],h)+b[t]))
    deltaq[0:t] = - ((1/Z)*np.exp(np.dot(w.T[0:t],h)+b[0:t])) 
    deltaq[t+1:len(deltaq)] = - ((1/Z)*np.exp(np.dot(w.T[t+1:len(deltaq)],h)+b[t+1:len(deltaq)]))  
    deltaq = np.array([deltaq]) 
    h = np.array([h]).T
    #h = np.asarray([h])
    #print  w.T.shape, h.shape, deltaq.shape
#    print deltaq.shape, w.shape
    #deltah = np.dot(np.dot(w.T,h),deltaq)
    deltah = np.dot(deltaq,w.T)
    '''
    deltah[t] = np.dot(w.T[t], np.exp(np.log(h))) - \
            (1/Z) * (np.exp(np.dot(w.T[t],h) + b[t])*np.dot(w.T[t], np.exp(np.log(h))))
    deltah[0:t] = -\
            (1/Z) * (np.exp(np.dot(w.T,h) + b)*np.dot(w.T, np.exp(np.log(h))))
    deltah[t+1:len(deltah)] = - \
            (1/Z) * (np.exp(np.dot(w.T,h) + b)*np.dot(w.T, np.exp(np.log(h))))
    '''
    x = np.array([x])
    
    #deltaq = np.array([deltaq])
    #deltah = np.array([deltah])
    
    #calculate derivatives according to a single datapoint
#    print deltaq.shape, h.shape
    gradw = np.dot(h,deltaq)
#    print gradw.shape
    #gradb = np.squeeze(np.asarray(deltaq))
    #print deltaq.shape
    gradb = np.reshape(deltaq,-1)
    #print gradb.shape
#    print gradb.shape
    #print np.dot(deltaq.T, deltah)
    #print 1- np.dot(h.T , x)
#    print "begin"
    #print deltaq.shape, deltah.shape, h.shape, x.shape, np.dot(h, x).shape 
    #print np.dot(deltaq.T,deltah.T).shape, np.dot(h, x).shape 
    I = np.matrix(np.identity(v.shape[1]))
    #print deltah.shape, h.T.shape, I.shape, (1-h).shape, I.shape, v.shape
    #gradv = np.dot(np.dot(np.dot(np.dot(np.dot(deltah.T,h.T),I),1-h),I),v).T
    #print deltah.T.shape, h.T.shape, I.shape, (1-h).shape, I.shape, v.shape
    #print np.diag(np.dot(np.dot(np.dot(deltah.T, h.T),I),h)).shape
    diagonal = np.dot(np.dot(np.dot(deltah.T, h.T),I),h)
    #print np.diag(np.squeeze(np.asarray(diagonal))).shape
    gradv = np.dot(v, np.diag(np.squeeze(np.asarray(diagonal))))    
    # V x diag(delta sigma^T I sigma)
    #gradv = np.dot(np.dot(deltaq, deltah), 1- np.dot(h.T , x))
#    print gradv.shape
    #grada = np.dot(np.dot(deltaq.T, deltah).T,1- h.T)
#    print deltah.shape, h.T.shape,(1-h).shape
    grada = np.dot(np.dot(deltah.T,h.T),1-h)
    #grada = np.squeeze(np.asarray(grada))
    grada = np.reshape(grada,-1)
    return gradw, gradb, gradv, grada
    
def return_likelihood_mlp(x, t, w, b, v, a):
   
   logq, logp = np.zeros(b.shape[0]), np.zeros(b.shape[0])  
   
   b = np.array([b,]*x.shape[0]).T
   
   h = calc_hidden(x, v, a)
   logq = np.dot(w.T, h) + b
    
   Z = np.sum(np.exp(logq), axis=0)
   Z = np.array([Z,]*b.shape[0])
   
   logp = logq - np.log(Z)
   
   return logp

w, b, v, a, logp_t, logp_v = train_mlp(x_train, t_train, w, b, v, a, 3)
plt.plot(logp_t, color = 'b', label = 'training')
plt.plot(logp_v, color = 'g', label = 'validation') 
plt.legend()

#print the final weights
plt.figure("weights")  
plot_digits(w.T, numcols=5)


#validate and plot images with the best and worst 8 probabilities
best, worst = validate(x_valid, t_valid, w, b, 8)
#print best
plot_res_reg(best, worst)

plt.show()

