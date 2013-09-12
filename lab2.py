# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:06:46 2013

@author: pathos
"""
from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
#import scipy as sp
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
    w = np.dot(pinv, np.transpose(np.dot(phi.transpose(), t)))
    return w.transpose(), phi

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
    #print valid_phi.shape
    #print w.shape
    #print valid_data_t.shape
    temp = np.dot(valid_phi,w.transpose()) - valid_data_t
    temp2 = np.dot(valid_phi,w.transpose()) - valid_data_t
    Ew = np.dot(np.transpose(temp),temp2)
    #print Ew
    temp3 = np.dot((lamd/2),w)
    temp3 = np.dot(temp3,w.transpose())
    Ew = Ew + temp3
    #print Ew
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
               #smooth, xpol = smoothing(M, 100)
               valid_phi = calculatePhi(valid_data, M)
               error = calculateError(w, valid_phi, valid_data_t, lamd)
               #error = np.mean(errors)
               #error_fold = []
               #error_fold.append(error)
               #error_fold.append(M)
               #error_fold.append(lamd)
               
               #wtest, phitest = fit_polynomial_reg(valid_data, valid_data_t, M, lamd)
               
#               print valid_data, valid_data_t
#               plt.plot(train_data, train_data_t, 'co')
#               print
#               minimize E(w) = 1/2(Phi*w - t).transpose()*(Phi*w - t) +(lamda/2)*w.transpose()*w
#               print "phi.shape: " + str(phi.shape)
#               print "w.shape: " + str(w.shape)
#               print "train_t shape: " + str(train_data_t.shape)
#               Ew = calculateError(w.transpose(), phi, train_data_t, lamd)
#               curve = np.dot(phi,w.transpose())
#               min_dist = calcError(curve, valid_data)
               errorS_fold.append(error) 
               #plt.show()
               #print "Ew.shape: " + str(Ew.shape)
               '''
               labelest = "M: "+str(M)+" lamda: "+str(lamd)
               plt.plot(xpol,np.sin(xpol), 'green', label = "original")
               plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = labelest)
               plt.legend()
               plt.show()
               '''
           #print errors_fold
           #print calcMSE(errors_fold)
           all_MSE_errors[M,l] = calcMSE(errorS_fold)
           #element = [calcMSE(errorS_fold), M, l]
           #all_MSE_errors.append(element)
    #plot_M_L_results(all_MSE_errors)
    bestM, bestL = findBestParameters(all_MSE_errors)
    print "bestM: " + str(bestM)+ " bestL: " + str(bestL)
    return bestM, bestL, all_MSE_errors
    #findBestParameters(all_MSE_errors)
    
def calcMSE(errorS_fold):
    return np.mean(errorS_fold)

def findBestParameters(all_MSE_errors):
    #print all_MSE_errors
    #print sorted(all_MSE_errors.iterkeys(), key=lambda k: all_MSE_errors[k][1])
    bestfit = min(all_MSE_errors, key=all_MSE_errors.get)

    #all_MSE_errors.sort(key = lambda x: x[0])
    #print all_MSE_errors
    #bestfit = all_MSE_errors[0]
    bestM = bestfit[0]
    bestL = bestfit[1]
    return bestM, bestL
    
def plot_M_lamda_error(all_MSE_errors):
    #print all_MSE_errors
    #ax = fig.add_subplot(111, projection='3d')
    #X = all_MSE_errors[]
    datapoints = []
    for item in all_MSE_errors.iteritems():
        datapoints.append((item[0][0],item[0][1],item[1]))
    #print datapoints
    x,y,z = zip(*datapoints)
    #X = [] 
    #Y = []
    #Z = []
    #print X
    #print Y
    #print Z
    '''
    for key in all_MSE_errors.keys():
        if key[0] not in X:
            X.append(key[0])
        if key[1] not in Y:
            Y.append(key[1])
    X.sort();
    Y.sort();
    for M in X:
        for lamd in Y:
            Z.append(all_MSE_errors[(M,lamd)])
    
    for M in X:
        for lamd in Y:
            for error in Z:
                print M,lamd,error 
    print X
    print Y
    print Z
    '''
    #tuple(X)
    #tuple(Y)
    #tuple(Z)
    #print X.shape
    #print Y.shape
    #print Z.shape
    #X, Y, Z = axes3d.get_test_data(0.05)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    zs = np.array([x**2 + y for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('M')
    ax.set_ylabel('lamda')
    ax.set_zlabel('Error')
    #ax = fig.gca(projection='3d')
    #ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
    #ax.plot(X, Y, Z)
    #ax.plot_trisurf(X, Y, Z)
    #plt.show()    

#generate data    
x,t = gen_sinusoidal(9)
x2 = x[0:9]
t2 = t[0:9]

plotunregularized = 0

if plotunregularized == 0:
    #for M = 0
    plt.figure(1)
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
    

plotregularized = 0

if plotregularized == 0:
    #define lamda
    lamd = np.exp(-8)
    #print lamd
    
    #regularized M = 0
    plt.figure(2)
    plt.subplot(221)
    M = 0
    w,phi = fit_polynomial_reg(x2, t2, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green', label = "orig")
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = "estimated_reg")
    
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

plt.figure(3)
#plt.subplot(224)
M = bestM
w,phi = fit_polynomial_reg(x, t, M, bestL)
smooth,xpol = smoothing(M, 100)
plt.plot(x,t,'co')
plt.plot(xpol,np.sin(xpol), 'green', label="original")
plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = "estimated")
plt.legend()

plot_M_lamda_error(all_MSE_errors)
#show figure
plt.show()
