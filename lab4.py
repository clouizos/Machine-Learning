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
    
def computec(K, beta):
    c = np.zeros((K.shape))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            c[i,j] = K[i,j] + (1/beta)
    return c
    
def gp_predictive_distribution( x_train, x_test, theta, C = None ):
    #k = k_n_m(x_train,x_train,theta)
    K = computeK(x_train, x_train, theta)
    mu = np.zeros((x_train.shape[0],x_test.shape[0]))
    var = np.zeros((x_train.shape[0], x_test.shape[0]))
    if C == None:
        C = computeC(K, beta)
    c = computec(K, beta)
    invC = np.linalg.inv(C)
    for i in range(len(x_test)):
    #print K.T.shape, invC.shape, x_test.shape
        mu[:,i] = np.dot(np.dot(K.T,invC),x_test[i])
        var[:,i] = c - mu
    
    return mu, var
    
def gp_log_likelihood(x_train, t_train, theta, C = None, invC = None):
    K = computeK(x_train, x_train, theta)
    if C == None:
        C = computeC(K, theta)
    if invC == None:
        invC = np.linalg.inv(C)
    
    log_like = -(1/2)*np.log(np.abs(C)) - (1/2)*np.dot(np.dot(t_train.T,invC),t_train) -\
                    (C.shape[0]/2)*np.log(2*np.pi)    
    return log_like
    
def gp_plot( x_test, y_test, mu_test, var_test, x_train, t_train, theta, beta ):
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
    
    plt.plot( x_test, y_test, 'b', lw=3)
    plt.plot( x_test, mu_test, 'k--', lw=2 )
    plt.fill_between( x_test, mu_test+2*std_combo,mu_test-2*std_combo, color='k', alpha=0.25 )
    plt.fill_between( x_test, mu_test+2*std_model,mu_test-2*std_model, color='r', alpha=0.25 )
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

for i in range(len(thetas)):
    mu_test, var_test = gp_predictive_distribution(x_train, x_test, thetas[i])
    print mu_test, var_test
    gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, thetas[i], beta)

plt.show()

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