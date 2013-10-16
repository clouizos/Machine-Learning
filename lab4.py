# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:20:16 2013

@author: pathos
"""

from __future__ import division
from operator import itemgetter
from scipy import optimize
import numpy as np
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
    return thetas[0]*np.exp((-thetas[1]/2)*(xn - xm)**2)+thetas[2]+thetas[3]*xn*xm
    
def computeK( X1, X2, thetas ):
    K = np.zeros((X1.shape[0],X2.shape[0]))
    #K = np.zeros((X1.shape[0],len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):            
            K[i,j] = k_n_m(X1[i],X2[j],thetas)
    return K

#thetas = (np.array([1.00,4.00,0.00,0.00]),np.array([9.00,4.00,0.00,0.00]),np.array([1.00,64.00,0.00,0.00]),
          #np.array([1,00,0.25,0.00,0.00]),np.array([1.00,4.00,10.00,0.00]),np.array([1.00,4.00,0.00,5.00]))
thetas = ([1.00,4.00,0.00,0.00],[9.00,4.00,0.00,0.00],[1.00,64.00,0.00,0.00],
          [1.00,0.25,0.00,0.00],[1.00,4.00,10.00,0.0],[1.00,4.00,0.00,5.00])

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
    #print(mu_test+2*np.sqrt(K.diagonal())).shape
    #print mu_test+2*np.sqrt(K.diagonal())
    #print K.diagonal()
    plt.fill_between(x_test,mu_test+2*np.sqrt(np.diag(K)),mu_test-2*np.sqrt(np.diag(K)),color = 'red',alpha=0.1)
    plt.xlim(-1,1)    
    plt.legend(loc = 2, prop = {'size': 6}) 
#plt.show()

def computeC(K, beta):
    C = np.zeros((K.shape))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i==j:
                C[i,j] = K[i,j] + (1/beta)
            else:
                C[i,j] = K[i,j]
    return C
    
def computec(x_test, theta, beta):
    c = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        c[i] = k_n_m(x_test[i], x_test[i], theta) + (1/beta)
    return c
    
def computek(x_train, x_test, theta):
    k = np.zeros((x_train.shape[0],x_test.shape[0]))
    for j in range(x_test.shape[0]):
        for i in range(x_train.shape[0]):
            k[i,j] = k_n_m(x_train[i],x_test[j],theta)
    return k
    
    
def gp_predictive_distribution( x_train, x_test, t_train, theta, C = None ):
    if C == None:
        K = computeK(x_train, x_train, theta)
        C = computeC(K, beta)
    invC = np.linalg.inv(C)
    k = computek(x_train,x_test, theta)
    c = computec(x_test, theta, beta)
    mu = np.dot(np.dot(k.T, invC), t_train)
    var = c - np.dot(np.dot(k.T, invC), k)
    mu = np.reshape(mu, (x_test.shape[0]))
    return mu, var
    
def gp_log_likelihood(x_train, t_train, theta, C = None, invC = None):
    K = computeK(x_train, x_train, theta)
    if C == None:
        C = computeC(K, beta)
    if invC == None:
        invC = np.linalg.inv(C)
    log_like = -(1/2)*np.log(np.linalg.det(C)) - (1/2)*np.dot(np.dot(t_train.T,invC),t_train) -(C.shape[0]/2)*np.log(2*np.pi)    
    return log_like
    
def gp_plot( x_test, y_test, mu_test, var_test, x_train, t_train, theta, beta, log_like, label = None ):
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
    log_like = "%.2f" % log_like
    if label == None:
        plt.plot( x_test, y_test, 'b', lw=3, label = str(theta)+str(" ")+str(log_like))
    else:
        plt.plot( x_test, y_test, 'b', lw=3, label = label+str(" ")+str(theta))
    plt.plot( x_test, mu_test, 'k--', lw=2 )
    plt.fill_between( x_test, mu_test+2*std_combo,mu_test-2*std_combo, color='k', alpha=0.25 )
    plt.fill_between( x_test, mu_test+2*std_model,mu_test-2*std_model, color='r', alpha=0.25 )
    plt.plot( x_train, t_train, 'ro', ms=10 )

def grid_search(x_train, t_train, thetas):
    results = []
    for theta in thetas:
        K = computeK(x_train, x_train, theta)
        C = computeC(K, beta)
        invC = np.linalg.inv(C)
        log_like = gp_log_likelihood(x_train, t_train, theta, C, invC)
        results.append((theta, log_like))
    results = sorted(results, key = itemgetter(1), reverse = True)
    return results
    
def create_grid_thetas(max1,max2,max3,max4):
    thetas = []
    step = 0.5
    step2 = 0.1
    max1 = np.arange(0,max1+step2,step2)
    max2 = np.arange(0,max2+step,step)
    max3 = np.arange(0,max3+step,step)
    max4 = np.arange(0,max4+step,step)
    for i in max1:
        for j in max2:
            for k in max3:
                for l in max4:
                    thetas.append((i,j,k,l))
    return thetas

# function to optimize  
def func_like(thetas):
    thetas = np.exp(thetas)
    return -gp_log_likelihood(x_train, t_train, thetas)

# derivative of that function
def func_like_prime(thetas):
     return np.array((np.exp((-thetas[1]/2)*(np.linalg.norm(x)))),(),(),())

# function that tries to learn the hyperparameters
def learn_hyperparameters(nr_hyper):
    print optimize.fmin_cg(func_like,0)
    
    
'''
training for 2 datapoints
'''    
N_train = 2
x_train = []
for i in range(N_train):
    x_train.append(np.random.uniform(-1,1))
x_train = np.squeeze(np.asarray(x_train))
mu_train = np.zeros(N_train)
y_train = true_mean_function(x_train)
t_train = add_noise(y_train, sigma)

plt.figure("predictive distribution with 2 training data points")
for i in range(len(thetas)):
    mu_test, var_test = gp_predictive_distribution(x_train, x_test, t_train, thetas[i])
    log_like = gp_log_likelihood(x_train, t_train, thetas[i])
    plt.subplot(2,3,i+1)
    gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, thetas[i], beta, log_like)
    plt.legend(loc = 3, prop = {'size': 6})
    plt.xlim(-1,1)

'''
training for 10 datapoints
'''
N_train = 10
x_train = []
for i in range(N_train):
    x_train.append(np.random.uniform(-1,1))
x_train = np.squeeze(np.asarray(x_train))
mu_train = np.zeros(N_train)
y_train = true_mean_function(x_train)
t_train = add_noise(y_train, sigma)

plt.figure("predictive distribution with 10 training data points")
for i in range(len(thetas)):
    mu_test, var_test = gp_predictive_distribution(x_train, x_test, t_train, thetas[i])
    log_like = gp_log_likelihood(x_train, t_train, thetas[i])
    plt.subplot(2,3,i+1)
    gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, thetas[i], beta, log_like)
    plt.legend(loc = 3, prop = {'size': 6})
    plt.xlim(-1,1)

#set maximum values for the grid search
#thetas_grid = create_grid_thetas(5.0,5.0,5.0,10.0)  
thetas_grid = create_grid_thetas(2.0,5.0,2.0,0.0) 
#thetas_grid = create_grid_thetas(2.0,15.0,5.0,1.0)
#thetas_grid = create_grid_thetas(1.0,1.0,1.0,1.0)
grid_res = grid_search(x_train, t_train, thetas_grid)
best = grid_res[0][0]
#print best
worst = grid_res[len(grid_res)-1][0]
#print grid search results
print "---------------------results from the the grid search-----------------------"
#print grid_res
print "----------------------------------------------------------------------------"

#plot best combination of thetas
plt.figure("best and worst thetas according to the grid search")
mu_test, var_test = gp_predictive_distribution(x_train, x_test, t_train, best)
log_like = grid_res[0][1]
plt.subplot(2,1,1)
label = 'best'
gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, best, beta, log_like, label)
plt.legend(loc = 2, prop = {'size': 6})
plt.xlim(-1,1)
#plt.ylim(-2,2)

#plot worst combination of thetas
mu_test, var_test = gp_predictive_distribution(x_train, x_test, t_train, worst)
log_like = grid_res[len(grid_res)-1][1]
plt.subplot(2,1,2)
label = 'worst'
gp_plot(x_test, y_test, mu_test, var_test, x_train, t_train, worst, beta, log_like, label)
plt.legend(loc = 2, prop = {'size': 6})
plt.xlim(-1,1)
#plt.ylim(-2,2)

learn_hyperparameters(4)
plt.show()