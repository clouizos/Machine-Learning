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
    C = np.zeros((X1.shape[0],X2.shape[0]))
    for i in range(len(X1)):
        for j in range(len(X2)):
            if X1[i] == X2[j]:
                C[i,j] = k_n_m(X1[i],X2[j],thetas) + beta**(-1)*(0)
            elif X1[i] != X2[j]:
                C[i,j] = k_n_m(X1[i],X2[j],thetas) + beta**(-1)*(1)
            #print C[i,j]
    return C

#thetas = (np.array([1.00,4.00,0.00,0.00]),np.array([9.00,4.00,0.00,0.00]),np.array([1.00,64.00,0.00,0.00]),
          #np.array([1,00,0.25,0.00,0.00]),np.array([1.00,4.00,10.00,0.00]),np.array([1.00,4.00,0.00,5.00]))
thetas = ([1.00,4.00,0.00,0.00],[9.00,4.00,0.00,0.00],[1.00,64.00,0.00,0.00],
          [1.00,0.25,0.00,0.00],[1.00,4.00,10.00,0.00],[1.00,4.00,0.00,5.00])
#X1 = np.linspace(-1,1,N_test)
#X2 = np.linspace(-1,1,N_test)
#if (X1 == x_test).all:
#    print 'equal'
plt.figure("samples from the Gaussian process prior")
for i in range(len(thetas)):
    C = computeK(x_test, x_test, thetas[i])
    y_i1 = np.random.multivariate_normal(mu_test, C)
    y_i2 = np.random.multivariate_normal(mu_test, C)
    y_i3 = np.random.multivariate_normal(mu_test, C)
    y_i4 = np.random.multivariate_normal(mu_test, C)
    y_i5 = np.random.multivariate_normal(mu_test, C)
    plt.subplot(2,3,i+1)
    plt.plot(x_test,y_i1,'green',label = str(thetas[i]))
    plt.plot(x_test,y_i2,'blue')
    plt.plot(x_test,y_i3,'purple')
    plt.plot(x_test,y_i4,'orange')
    plt.plot(x_test,y_i5,'black')
    #y = generate_t(x_test, sigma);
    #plt.plot(x_test,y_test,'r--',label='true mean')
    #plt.plot()
    #plt.plot(x_test,mu_test,'gray',label = 'true mean')
    #plt.plot(x_test, mu_test+2*np.sqrt(C), 'r--', label = 'uncertainty')
    #plt.plot(x_test, mu_test-2*np.sqrt(C), 'r--')
    #plt.fill_between(x_test,mu_test+2*np.sqrt(C),mu_test-2*np.sqrt(C),color = 'red',alpha=0.1)
    plt.legend(loc = 2, prop = {'size': 6}) 
plt.show()