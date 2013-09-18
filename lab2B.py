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
    
x,t = gen_sinusoidal2(9)
print x
print
print t