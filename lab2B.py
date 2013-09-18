# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:00:08 2013

@author: pathos
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def gen_sinusoidal2(N):
    #x = np.linspace(0, 2*np.pi, N, endpoint=True)
    x = np.random.uniform(0,2*np.pi,N)
    mu, sigma = np.sin(x), 0.2
    t = np.random.normal(mu, sigma, N)
    return x,t
    
def getPhi(x, M):
    phi = np.mat(np.zeros((x.shape[0],M+1)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j==0:
                phi[i,j] = 1
            phi[i,j] = np.power(x[i],j)
    return phi
    
def fit_polynomial_bayes(x, t, M, alpha, beta):
    phi = getPhi(x, M)
    identity = np.identity(M+1)
    aIden = np.dot(alpha, identity)
    betaPhi = np.dot(np.dot(beta, phi.transpose()), phi)
    Sn = np.linalg.inv(aIden + betaPhi)
    mn = np.dot(np.dot(np.dot(beta, Sn), phi.transpose()), t)
    return Sn, mn

def predict_polynomial_bayes(x, m, S, beta):
    phi = getPhi(x, S.shape[0]-1)
    predS = 1/beta + np.dot(np.dot(phi, S), phi.transpose())
    predM = np.dot(m, phi.transpose())
    return predS, predM
    
x,t = gen_sinusoidal2(7)
alpha = 1/2
beta = 1/(0.2)**2
M = 5
Sn, mn = fit_polynomial_bayes(x, t, M, alpha, beta)
prediction_points = 10
xforPred = np.linspace(0, 2*np.pi, prediction_points, endpoint = True)
predS, predM = predict_polynomial_bayes(xforPred, mn, Sn, beta)
phiPred = getPhi(xforPred, M+1)
print predM.shape
print phiPred.shape
plt.figure("predictive distribution")
plt.plot(xforPred, 1/(np.dot(predS, np.sqrt(2 * np.pi))) *
             np.exp( - np.dot((phiPred - predM), 2) / np.dot(2, np.dot(predS, 2))),
              linewidth=2, color='r')
#plt.fill_between(,alpha = 0.1)
plt.show()

