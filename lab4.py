# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:20:16 2013

@author: pathos
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sigma = 0.1
beta  = 1.0 / pow(sigma,2) # this is the beta used in Bishop Eqn. 6.59
N_test = 100
x_test = np.linspace(-1,1,N_test); 
mu_test = np.zeros( N_test )

def true_mean_function( x ):
    return np.sin( 2*np.pi*(x+1) )

def add_noise( y, sigma ):
    return y + sigma*np.random.randn(len(y))

def generate_t( x, sigma ):
    return add_noise( true_mean_function( x), sigma )
    
y_test = true_mean_function( x_test )
t_test = add_noise( y_test, sigma )
#plt.plot( x_test, y_test, 'b-', lw=2)
#plt.plot( x_test, t_test, 'go')

def k_n_m( xn, xm, thetas ):
    return thetas[0]*np.exp((-thetas[1]/2)*np.linalg.norm(xn - xm)**2)+thetas[2]+thetas[3]*xn*xm
    
def computeK( X1, X2, thetas ):
    K = np.zeros((X1.shape[0],X2.shape[0]))
    #K = np.zeros((X1.shape[0],len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):            
            #if i == j:
                #K[i,j] = k_n_m(X1[i],X2[j],thetas) + beta**(-1)*(1)
            K[i,j] = k_n_m(X1[i],X2[j],thetas)
            #else:
                #K[i,j] = k_n_m(X1[i],X2[j],thetas) + beta**(-1)*(0)            
            #C[i,j] = k_n_m(X1[i],X2[j],thetas) + beta**(-1)*(X1[i]-X2[j])
    return K

#thetas = (np.array([1.00,4.00,0.00,0.00]),np.array([9.00,4.00,0.00,0.00]),np.array([1.00,64.00,0.00,0.00]),
          #np.array([1,00,0.25,0.00,0.00]),np.array([1.00,4.00,10.00,0.00]),np.array([1.00,4.00,0.00,5.00]))
thetas = ([1.00,4.00,0.00,0.00],[9.00,4.00,0.00,0.00],[1.00,64.00,0.00,0.00],
          [1.00,0.25,0.00,0.00],[1.00,4.00,10.00,0.00],[1.00,4.00,0.00,5.00])

plt.figure("samples from the Gaussian process prior")
for i in range(len(thetas)):
    K = computeK(x_test, x_test, thetas[i])
    y_i1 = np.random.multivariate_normal(mu_test, K)
    y_i2 = np.random.multivariate_normal(mu_test, K)
    y_i3 = np.random.multivariate_normal(mu_test, K)
    y_i4 = np.random.multivariate_normal(mu_test, K)
    y_i5 = np.random.multivariate_normal(mu_test, K)
    plt.subplot(2,3,i+1)
    plt.plot(x_test,y_i1,'green',label = str(thetas[i]))
    plt.plot(x_test,y_i2,'blue')
    plt.plot(x_test,y_i3,'purple')
    plt.plot(x_test,y_i4,'orange')
    plt.plot(x_test,y_i5,'black')
    plt.plot(x_test,mu_test,'r--',label = 'true mean')
    plt.fill_between(x_test,mu_test+2*np.sqrt(np.diag(K)),mu_test-2*np.sqrt(np.diag(K)),color = 'red',alpha=0.1)
    plt.xlim(-1,1)    
    plt.legend(loc = 2, prop = {'size': 6}) 
#plt.show()

def computeC(K, beta):
    #print K.shape
    C = np.zeros((K.shape))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i==j:
                C[i,j] = K[i,j] + (1/beta)
            else:
                C[i,j] = K[i,j]
    return C
    
def computec(xn, xm, theta, beta):
    '''
    c = np.zeros((K.shape))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            c[i,j] = K[i,j] + (1/beta)
    '''
    return k_n_m(xn, xm, theta) + (1/beta)
    
def computek(x_train, x, theta):
    k = np.zeros(x_train.shape[0])
    #K = np.zeros((X1.shape[0],len(X2)))
    for i in range(k.shape[0]):
        k[i] = k_n_m(x_train[i],x,theta)
    return np.reshape(k, (k.shape[0],-1))
    
    
def gp_predictive_distribution( x_train, x_test, theta, C = None ):
    #k = k_n_m(x_train,x_train,theta)
    if C == None:
        K = computeK(x_train, x_train, theta)
        C = computeC(K, beta)
    invC = np.linalg.inv(C)
    #k = computeK(x_train, x_test, theta)
    #mu = np.zeros((x_train.shape[0],x_test.shape[0]))
    mu = np.zeros(x_test.shape[0])
    #var = np.zeros((x_train.shape[0], x_test.shape[0]))
    var = np.zeros(x_test.shape[0])
    for i in range(len(x_test)):
        c = computec(x_test[i],x_test[i], theta, beta)
        #k = computeK(x_train, np.asarray(x_test[i]), theta)
        k = computek(x_train, x_test[i], theta)
        #print k.T.shape, invC.shape,x_test[i]
        #print np.dot(k.T, invC).shape, x_train.shape
        mu[i] = np.dot(np.dot(k.T,invC), x_train)
        var[i] = c - np.dot(np.dot(k.T,invC),k)
    '''
    if C == None:
        C = computeC(K, beta)
    c = computec(K, beta)
    print c.shape
    invC = np.linalg.inv(C)
    #for i in range(len(x_test)):
    print k.T.shape, invC.shape, x_test.shape
    mu = np.dot(np.dot(k.T,invC).T,x_test)
    var = c - mu
    '''
    return mu, var
    
def gp_log_likelihood(x_train, t_train, theta, C = None, invC = None):
    K = computeK(x_train, x_train, theta)
    if C == None:
        C = computeC(K, beta)
    if invC == None:
        invC = np.linalg.inv(C)
    
    log_like = -(1/2)*np.log(np.linalg.norm(C)) - (1/2)*np.dot(np.dot(t_train.T,invC),t_train) -(C.shape[0]/2)*np.log(2*np.pi)    
    return log_like
    
def gp_plot( x_test, y_test, mu_test, var_test, x_train, t_train, theta, beta, log_like ):
    # x_test: 
    # y_test:   the true function at x_test
    # mu_test:   predictive mean at x_test
    # var_test: predictive covariance at x_test 
    # t_train:  the training values
    # theta:    the kernel parameters
    # beta:      the precision (known)
    
    # the reason for the manipulation is to allow plots separating model and data stddevs.
    std_total = np.sqrt(np.diag(var_test))         # includes all uncertainty, model and target noise 
    std_model = np.sqrt( std_total**2 - 1.0/beta ) # remove data noise to get model uncertainty in stddev
    std_combo = std_model + np.sqrt( 1.0/beta )    # add stddev (note: not the same as full)
    
    plt.plot( x_test, y_test, 'b', lw=3, label = str(theta)+str(" ")+str(log_like))
    plt.plot( x_test, mu_test, 'k--', lw=2 )
    plt.fill_between( x_test, (mu_test+2*std_combo).reshape(-1),(mu_test-2*std_combo).reshape(-1), color='k', alpha=0.25 )
    plt.fill_between( x_test, (mu_test+2*std_model).reshape(-1),(mu_test-2*std_model).reshape(-1), color='r', alpha=0.25 )
    plt.plot( x_train, t_train, 'ro', ms=10 )


'''
training for 2 datapoints
'''    
N_train = 2
#sigma = 0.1
#beta = 1.0/(sigma**2)
x_train = np.linspace(-1, 1, N_train)
mu_train = np.zeros(N_train)
y_train = true_mean_function(x_train)
t_train = add_noise(y_train, sigma)

plt.figure("predictive distribution with 2 training data points")
for i in range(len(thetas)):
    mu_test, var_test = gp_predictive_distribution(x_train, x_test, thetas[i])
    #print mu_test.shape, var_test.shape
    mu_test = mu_test.reshape((mu_test.shape[0],-1))
    var_test = var_test.reshape((var_test.shape[0], -1))
    log_like = gp_log_likelihood(x_train, t_train, thetas[i])
    #print log_like
    #print mu_test.shape, var_test.shape
    plt.subplot(2,3,i+1)
    gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, thetas[i], beta, log_like)
    plt.legend(loc = 2, prop = {'size': 6})
    plt.xlim(-1,1)

'''
training for 10 datapoints
'''
N_train = 10
#sigma = 0.1
#beta = 1.0/(sigma**2)
x_train = np.linspace(-1, 1, N_train)
mu_train = np.zeros(N_train)
y_train = true_mean_function(x_train)
t_train = add_noise(y_train, sigma)

plt.figure("predictive distribution with 10 training data points")
for i in range(len(thetas)):
    mu_test, var_test = gp_predictive_distribution(x_train, x_test, thetas[i])
    #print mu_test.shape, var_test.shape
    mu_test = mu_test.reshape((mu_test.shape[0],-1))
    var_test = var_test.reshape((var_test.shape[0], -1))
    log_like = gp_log_likelihood(x_train, t_train, thetas[i])
    #print log_like
    #print mu_test.shape, var_test.shape
    plt.subplot(2,3,i+1)
    gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, thetas[i], beta, log_like)
    plt.legend(loc = 2, prop = {'size': 6})
    plt.xlim(-1,1)
    
plt.show()