# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:00:08 2013

@author: pathos
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def gen_sinusoidal2(N):
    x = np.random.uniform(0,2*np.pi,N)
    mu, sigma = np.sin(x), 0.2
    t = np.random.normal(mu, sigma, N)
    return x,t

def smoothing(M, count):
    x = np.linspace(0, 2*np.pi,count, endpoint = True)
    phi = np.mat(np.zeros((x.shape[0],M+1)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j==0:
                phi[i,j] = 1
            phi[i,j] = np.power(x[i],j)
    return phi,x
    
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
    
def fit_polynomial_bayes(x, t, M, alpha, beta):
    phi = getPhi(x, M)
    identity = np.identity(M+1)
    aIden = np.dot(alpha, identity)
    betaPhi = np.dot(np.dot(beta, phi.transpose()), phi)
    Sn = np.linalg.inv(aIden + betaPhi)
    mn = np.dot(np.dot(np.dot(beta, Sn), phi.transpose()), t)
    mn = mn.T
    return mn, Sn

def predict_polynomial_bayes(x, m, S, beta):
    phi = getPhi(x, S.shape[0]-1).T
    predS = 1/beta + np.dot(np.dot(phi.T, S), phi)
    predM = np.dot(m.T, phi)
    return predM, predS



x, t = gen_sinusoidal2(7)
alpha = 1/2
beta = 1/(0.2)**2
M = 5
phi = getPhi(x, M)
mn, Sn = fit_polynomial_bayes(x, t, M, alpha, beta)

xPred = np.linspace(0,2*np.pi,100)
predM = np.zeros(len(xPred))
predS = np.zeros(len(xPred))
for i in range(len(xPred)):
    predM[i], predS[i] = predict_polynomial_bayes(xPred[i], mn, Sn, beta)

plt.figure("bayesian linear regression")
plt.subplot(211)
plt.plot(x,t, 'co')
plt.plot(xPred, predM,'green', label = 'mean')
plt.plot(xPred, predM+np.sqrt(predS), 'r--', label = 'uncertainty')
plt.plot(xPred, predM-np.sqrt(predS), 'r--')
plt.fill_between(xPred,predM+np.sqrt(predS),predM-np.sqrt(predS),color = 'red',alpha=0.1)
plt.xlim([0,2*np.pi])
plt.legend()

plt.subplot(212)
arrayM = np.squeeze(np.array(mn))
wPost = np.random.multivariate_normal(arrayM,Sn, 100)
for i in range(len(wPost)):
    wPostSin = wPost[i]
    W = wPostSin.reshape(1,6)
    smooth,xpol = smoothing(M, 100)
    if i == 1:
        plt.plot(xpol, np.dot(smooth, W.T),'r', label = 'polynomials')
    else:
        plt.plot(xpol, np.dot(smooth, W.T),'r')
plt.plot(x,t, 'co')        
plt.xlim([0,2*np.pi])
plt.legend()

plt.show()