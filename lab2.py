# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:06:46 2013

@author: pathos
"""
from __future__ import division
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D, proj3d
import pylab
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt


def gen_sinusoidal(N):
    x = np.linspace(0, 2*np.pi, N, endpoint=True)
    mu, sigma = np.sin(x), 0.2
    t = np.random.normal(mu, sigma, N)
    return x,t

def fit_polynomial(x, t, M):
    phi = getPhi(x, M)
    pinverse = np.linalg.pinv(phi)
    w = np.dot(pinverse, t)
    return w, phi
    
def plot_unregularized(x, t, M_list):
    plt.figure("unregularized linear regression")
    for (M, i) in zip(M_list, range(len(M_list))):
        subplot = 220 + i+1
        plt.subplot(subplot)
        w,phi = fit_polynomial(x, t, M)
        smooth,xpol = smoothing(M, 100)
        plt.plot(x,t,'co')
        plt.plot(xpol,np.sin(xpol), 'green', label = "original")
        label = "M:"+str(M)
        plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = label)
        plt.legend()
    
def fit_polynomial_reg(x, t, M, lamd):
    phi = getPhi(x, M)
    pinv = np.dot(lamd, np.identity(M+1))      
    pinv = pinv + np.dot(phi.transpose(), phi)
    pinv = np.linalg.inv(pinv)
    w = np.dot(pinv, phi.transpose())
    w = np.dot(w,t)
    return w, phi
    
def getPhi(x, M):
    try: #x not a scalar value
        phi = np.mat(np.zeros((x.shape[0],M+1)))
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                if j==0:
                    phi[i,j] = 1
                phi[i,j] = np.power(x[i],j)
    except: #scalar value
        phi = np.zeros(M+1)
        phi = phi.T
        for i in range(phi.shape[0]):
            if i==0:
                phi[i] = 1
            phi[i] = np.power(x, i)
    return phi

def smoothing(M, count):
    '''
    function that creates a smoother plot since it creates a phi from more
    data points
    '''
    x = np.linspace(0, 2*np.pi,count, endpoint = True)
    phi = np.mat(np.zeros((x.shape[0],M+1)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j==0:
                phi[i,j] = 1
            phi[i,j] = np.power(x[i],j)
    return phi,x

def calculateError(w, valid_phi, valid_data_t, lamd):
    #calculate the error of the regression
    temp = np.dot(valid_phi,w.transpose()) - valid_data_t
    temp2 = np.dot(valid_phi,w.transpose()) - valid_data_t
    Ew = np.dot(np.transpose(temp),temp2)
    temp3 = np.dot((lamd/2),w)
    temp3 = np.dot(temp3,w.transpose())
    Ew = Ew + temp3
    return Ew.item(0)
   

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
               valid_phi = getPhi(valid_data, M)
               error = calculateError(w, valid_phi, valid_data_t, lamd)
               #make the list with the errors for each fold
               errorS_fold.append(error) 
           
           #append to the dictionary of errors the error for M,lamda 
           all_MSE_errors[M,l] = calcMSE(errorS_fold)
    #find the best M and lamda
    bestM, bestL = findBestParameters(all_MSE_errors)
    
    return bestM, bestL, all_MSE_errors
    
def calcMSE(errorS_fold):
    #return appropriate error so the visualization is clearer
    return np.log(np.mean(errorS_fold,dtype=np.float64))

def findBestParameters(all_MSE_errors):
    #find the best M and lambda according to the dictionary of the errors
    bestfit = min(all_MSE_errors, key=all_MSE_errors.get)
    bestM = bestfit[0]
    bestL = bestfit[1]
    return bestM, bestL
    
def plot_M_lamda_error(all_MSE_errors, bestM, bestL):
    datapoints = []
    bestError = all_MSE_errors[bestM, bestL]
    
    for item in all_MSE_errors.iteritems():
        #create tuples of data points where x = M, y = lambda, z = error
        datapoints.append((item[0][0],item[0][1],item[1]))
    
    #create the vector of x,y,z    
    x,y,z = zip(*datapoints)
    #create a flat surface
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    # interpolate for missing values 
    #and use zi as the height of the surface in specific points
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    xim, yim = np.meshgrid(xi, yi)
    
    bestxi = np.linspace(bestM, bestM, 1)
    bestyi = np.linspace(bestL, bestL, 1)

    fig = plt.figure("Error for different M and lamda")

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xim, yim, zi, cmap = cm.coolwarm)
    ax.plot(bestxi, bestyi, bestError, 'ro', label = 'minimum error')
    text='[M:'+str(int(bestM))+', lamda:'+str(int(bestL))+', error:'+str("%.2f" % round(bestError,2))+']'  
    x2, y2, _ = proj3d.proj_transform(bestM,bestL,bestError, ax.get_proj())
    pylab.annotate(text,
                       xycoords='data',
                       xy = (x2, y2), xytext = (0, 0),
                       textcoords = 'offset points', ha = 'right', va = 'bottom',
                       bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
    ax.set_xlabel('M')
    ax.set_ylabel('lamda')
    ax.set_zlabel('Error')
    ax.view_init(5,-110)
    ax.legend()    
    
#generate data    
x,t = gen_sinusoidal(9)
M_list = (0,1,3,8)
plot_unregularized(x, t, M_list)

#perform the cross validation and plot the best result and the error plot
train_folds,valid_folds = kfold_indices(len(x), 9)
bestM, bestL, all_MSE_errors = cross_validation(x, t, train_folds, valid_folds)
plot_M_lamda_error(all_MSE_errors, bestM, bestL)

'''
plotregularized = 0

if plotregularized == 0:
    #define lamda
    lamd = bestL
    #print lamd
    
    #regularized M = 0
    plt.figure("regularized linear regression with lamda exp(-"+str(lamd)+")")
    plt.subplot(221)
    M = 0
    w,phi = fit_polynomial_reg(x, t, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green', label = "original")
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = "estimated")
    plt.legend()
    
    #regularized M = 1
    #plt.figure(2)
    plt.subplot(222)
    M = 1
    w,phi = fit_polynomial_reg(x, t, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
    #regularized M = 3
    #plt.figure(2)
    plt.subplot(223)
    M = 6
    w,phi = fit_polynomial_reg(x, t, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
    
    #regularized M = 9
    #plt.figure(2)
    plt.subplot(224)
    M = 8
    w,phi = fit_polynomial_reg(x, t, M, lamd)
    smooth,xpol = smoothing(M, 100)
    plt.plot(x,t,'co')
    plt.plot(xpol,np.sin(xpol), 'green')
    plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue')
'''    

plt.figure("best model according to cross-validation")
labelML = "best M: " + str(bestM)+ " best lamda: exp(-" + str(bestL)+")"
M = bestM
w,phi = fit_polynomial_reg(x, t, M, bestL)
smooth,xpol = smoothing(M, 100)
plt.plot(x,t,'co')
plt.plot(xpol,np.sin(xpol), 'green', label="original")
plt.plot(xpol, np.dot(smooth, w.transpose()), 'blue', label = labelML)
plt.legend()

#show figure
plt.show()
