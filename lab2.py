# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:06:46 2013

@author: pathos
"""
from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def gen_sinusoidal(N):
    x = np.linspace(0, 2*np.pi, N, endpoint=True)
    mu, sigma = np.sin(x), 0.2
    t = np.random.normal(mu, sigma, N)
    return x,t

def fit_polynomial(x, t, M):
    phi = np.mat(np.zeros((x.shape[0],M+1)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j==0:
                phi[i,j] = 1
            phi[i,j] = np.power(x[i],j)
    pinverse = np.linalg.pinv(phi)
    w = np.dot(pinverse, t)
    return w, phi
    
def fit_polynomial_reg(x, t, M, lamd):
    phi = np.mat(np.zeros((x.shape[0],M+1)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j==0:
                phi[i,j] = 1
            phi[i,j] = np.power(x[i],j)
    pinv = np.dot(lamd, np.identity(M+1))      
    pinv = pinv + np.dot(phi.transpose(), phi)
    pinv = np.linalg.inv(pinv)
    w = np.dot(pinv, phi.transpose())
    w = np.dot(w,t)
    return w, phi

def smoothing(M, count):
    #print x.shape[0]
    x = np.linspace(0, 2*np.pi,count, endpoint = True)
    phi = np.mat(np.zeros((x.shape[0],M+1)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j==0:
                phi[i,j] = 1
            phi[i,j] = np.power(x[i],j)
    return phi,x

def calculateError(w, valid_phi, valid_data_t, lamd):
    temp = np.dot(valid_phi,w.transpose()) - valid_data_t
    temp2 = np.dot(valid_phi,w.transpose()) - valid_data_t
    Ew = np.dot(np.transpose(temp),temp2)
    temp3 = np.dot((lamd/2),w)
    temp3 = np.dot(temp3,w.transpose())
    Ew = Ew + temp3
    return Ew

def calculatePhi(valid_data, M):
    phi = np.mat(np.zeros((valid_data.shape[0],M+1)))
    #print phi
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            phi[i, j] = np.power(valid_data[i], j)
    return phi

def kfold_indices(N, k):
    all_indices = np.arange(N,dtype=int)
    np.random.shuffle(all_indices)
    idx = np.floor(np.linspace(0,N,k+1))
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold+1]]
        valid_folds.append(valid_indices)
        train_folds.append(np.setdiff1d(all_indices, valid_indices))
    return train_folds, valid_folds
    

def cross_validation(x, t, train_folds, valid_folds):
    all_MSE_errors = {}
    #iterate over possible M and lamda
    for M in range(11):
        for l in range(10, -1, -1):
           #print "M,l: "+ str((M,l))
           lamd = np.exp(-l)
           errorS_fold = []
           #get folds
           for (train_fold, valid_fold) in (zip(train_folds, valid_folds)):
               #initialize training and testing data
               train_data = np.zeros(train_fold.size)
               train_data_t = np.zeros(train_fold.size)
               valid_data = np.zeros(valid_fold.size)
               valid_data_t = np.zeros(valid_fold.size)
               #make training set
               for (i, index) in (zip(range(train_data.size), train_fold)):
                   train_data[i] = x[index]
                   train_data_t[i] = t[index]
               #make test set
               for (i, index) in (zip(range(valid_data.size), valid_fold)):
                   valid_data[i] = x[index]
                   valid_data_t[i] = t[index]
               #make the model based on training data
               
               w, phi = fit_polynomial_reg(train_data, train_data_t, M, lamd)
               valid_phi = calculatePhi(valid_data, M)
               error = calculateError(w, valid_phi, valid_data_t, lamd)
               
               errorS_fold.append(error) 
               
           all_MSE_errors[M,l] = calcMSE(errorS_fold)

    #plot_M_L_results(all_MSE_errors)
    bestM, bestL = findBestParameters(all_MSE_errors)
    #print "bestM: " + str(bestM)+ " bestL: " + str(bestL)
    return bestM, bestL, all_MSE_errors
    #findBestParameters(all_MSE_errors)
    
def calcMSE(errorS_fold):
    #return appropriate error so the visualization is clearer
    return np.log(np.mean(errorS_fold))

def findBestParameters(all_MSE_errors):
    bestfit = min(all_MSE_errors, key=all_MSE_errors.get)
    bestM = bestfit[0]
    bestL = bestfit[1]
    return bestM, bestL
    
def plot_M_lamda_error(all_MSE_errors):
    datapoints = []
    for item in all_MSE_errors.iteritems():
        datapoints.append((item[0][0],item[0][1],item[1]))
    #print datapoints
    x,y,z = zip(*datapoints)
    
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    xim, yim = np.meshgrid(xi, yi)
    
    #zi = griddata(x, y, z, xi, yi)
    fig = plt.figure("Error for different M and lamda")

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xim, yim, zi)

    
    ax.set_xlabel('M')
    ax.set_ylabel('lamda')
    ax.set_zlabel('Error')
    

#generate data    
x,t = gen_sinusoidal(9)
x2 = x[0:9]
t2 = t[0:9]

plotunregularized = 0

if plotunregularized == 0:
    #for M = 0
    plt.figure("unregularized linear regression")
    plt.subplot(221)
    M = 0
    w,phi = fit_polynomial(x2, t2, M)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green', label = "original")
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = "estimated")
    plt.legend()
    #plt.legend([p], ['original'])
    
    #for M = 1
    plt.figure(1)
    plt.subplot(222)
    M = 1
    w,phi = fit_polynomial(x2, t2, M)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
    #for M = 3
    plt.figure(1)
    plt.subplot(223)
    M = 3
    w,phi = fit_polynomial(x2, t2, M)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
    #for M = 9
    plt.figure(1)
    plt.subplot(224)
    M = 9
    w,phi = fit_polynomial(x2, t2, M)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    plt.legend()
    

plotregularized = 0

if plotregularized == 0:
    #define lamda
    lamd = np.exp(-5)
    #print lamd
    
    #regularized M = 0
    plt.figure("regularized linear regression with lamda exp(-5)")
    plt.subplot(221)
    M = 0
    w,phi = fit_polynomial_reg(x2, t2, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green', label = "original")
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = "estimated")
    plt.legend()
    
    #regularized M = 1
    plt.figure(2)
    plt.subplot(222)
    M = 1
    w,phi = fit_polynomial_reg(x2, t2, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
    #regularized M = 3
    plt.figure(2)
    plt.subplot(223)
    M = 3
    w,phi = fit_polynomial_reg(x2, t2, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
    #regularized M = 9
    plt.figure(2)
    plt.subplot(224)
    M = 9
    w,phi = fit_polynomial_reg(x2, t2, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
train_folds,valid_folds = kfold_indices(9, 9)
bestM, bestL, all_MSE_errors = cross_validation(x, t, train_folds, valid_folds)

plt.figure("best model according to cross-validation")
#plt.subplot(224)
labelML = "best M: " + str(bestM)+ " best lamda: " + str(bestL)
M = bestM
w,phi = fit_polynomial_reg(x, t, M, bestL)
smooth,xpol = smoothing(M, 100)
plt.plot(x,t,'co')
plt.plot(xpol,np.sin(xpol), 'green', label="original")
plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = labelML)
plt.legend()

plot_M_lamda_error(all_MSE_errors)
#show figure
plt.show()
