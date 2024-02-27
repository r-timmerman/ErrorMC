#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Provides method to sample (skewed) Gaussian distriutions
By Roland Timmerman
27 February 2024

Using the approximation for the cdf of a skew normal distribution
provided by Amsler et al. (2021), doi:10.1007/s00181-020-01868-6
"""


#Imports
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as st
from scipy.optimize import newton

def pdf_normal(x, mu=0, sigma=1):
    """
    Calculates the probability distribution function of a normal distribution
    
    Parameters
    ---------------
        x: float
            Sample position parameter
        mu: float
            Central position of distribution
        sigma: float
            Standard deviation of normal distribution
            
    Return
    ---------------
        pdf: float
            Value of pdf at position x
    """
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def cdf_normal(x, mu=0, sigma=1):
    """
    Calculates the cumulative distribution function of a normal distribution
    
    Parameters
    ---------------
        x: float
            Sample position parameter
        mu: float
            Central position of distribution
        sigma: float
            Standard deviation of normal distribution
            
    Return
    ---------------
        cdf: float
            Value of cdf at position x
    """
    return 0.5*(1+erf((x-mu)/(sigma*np.sqrt(2))))

def APS_UT(Q):
    """
    Calculates the approximate value of the skewed cdf in the upper tail
    
    Parameters
    ---------------
        Q: float
            Sample position parameter
            
    Return
    ---------------
        cdf: float
            Value of cdf at position Q
    """
    return 2*cdf_normal(Q) - 1

def APS_LT(Q, lambd=1):
    """
    Calculates the approximate value of the skewed cdf in the lower tail
    
    Parameters
    ---------------
        Q: float
            Sample position parameter
        lambd: float
            Skewness parameter
            
    Return
    ---------------
        cdf: float
            Value of cdf at position Q
    """
    return 2*cdf_normal(Q)*cdf_normal(lambd*Q)/(1+lambd**2)

def cdf_sn_central(Q, lambd=1):
    """
    Calculates the approximate value of the skewed cdf in the central region
    
    Parameters
    ---------------
        Q: float
            Sample position parameter
        lambd: float
            Skewness parameter
            
    Return
    ---------------
        cdf: float
            Value of cdf at position Q
    """
    rho = -lambd/np.sqrt(1+lambd**2)
    biv_normal = st.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]])
    if hasattr(Q, '__iter__'):
        output = np.array([2*biv_normal.cdf(np.array([q,0])) for q in Q])
    else:
        output = 2*biv_normal.cdf(np.array([Q,0]))
    return output

def find_mode_cdf(x_array, lambd):
    """
    Calculates the mode (most likely value) of a skewed normal distribution
    within the x-array range, assuming a skewness parameter (lambda), and
    assuming the mode falls within the range sampled by x_array
    
    Parameters
    ---------------
        x_array: array-like, dtype=float
            Sample positions to consider
        lambd: float
            Skewness parameter
            
    Return
    ---------------
        mode: float
            Mode of skewed normal distribution
    """
    cdf_array = cdf_sn_central(x_array, lambd=lambd)
    inverse_cdf = interpolate.interp1d(cdf_array, x_array, kind='linear')
    diff_array = cdf_array[1:]-cdf_array[:-1]
    peak = np.argmax(diff_array)
    return x_array[peak]

def test_func(lambd, plus_min_ratio):
    """
    Test function used by the optimization routine to estimate which
    value of lambd provides the requested ratio between the positive
    and negative uncertainties
    
    Parameters
    ---------------
        lambd: float
            Test skewness parameter
        plus_min_ratio: float
            (Absolute) target ratio between positive and negative errors
            
    Return
    ---------------
        diff: float
            Difference between input ratio and lambd-defined ratio
    """
    dx = 0.001
    x_array = np.arange(-13, 13, dx)
    cdf_array = cdf_sn_central(x_array, lambd=lambd)
    inverse_cdf = interpolate.interp1d(cdf_array, x_array, kind='linear')
    approx_mode = find_mode_cdf(x_array, lambd)
    
    x_prec = np.arange(approx_mode-5*dx, approx_mode+5*dx, dx/100)
    mode = find_mode_cdf(x_prec, lambd)
    
    upper = inverse_cdf(0.5+0.682689492/2)
    lower = inverse_cdf(0.5-0.682689492/2)
    return plus_min_ratio - (upper-mode)/(mode-lower)

def gaussian(mu, sigma_plus, sigma_minus=None, N=int(1e8)):
    """
    Provides an N-number of samples according to a (skewed) normal distribution.
    If one value of the uncertainty is given, a (symmetric) normal distribution
    is sampled N times, whereas if two uncertainties are given, a skewed
    normal distribution is sampled.
    
    Parameters
    ---------------
        mu: float
            Mode of distribution function
        sigma_plus: float
            (Positive) error
        sigma_minus: float (optional)
            Negative error
        N: int (optional)
            Number of samples requested
            
    Return
    ---------------
        samples: array
            Array of N samples drawn according to the requested distribution
    """
    if sigma_minus is None:
        return np.random.normal(mu, sigma_plus, N)
    else:
        #Find skewness that gives correct ratio between upper and lower limits
        plus_min_ratio = sigma_plus/sigma_minus
        lambd_best = newton(test_func, 1, args=(plus_min_ratio,), tol=1e-5)
        x_array = np.arange(-13, 13, 0.001)
        cdf_array = cdf_sn_central(x_array, lambd=lambd_best)
        
        #Apply approximations in the tails
        low_tail = np.where(cdf_array<1e-20)
        high_tail = np.where(cdf_array>1-1e-20)
        cdf_array[low_tail] = APS_LT(x_array[low_tail], lambd=lambd_best)
        cdf_array[high_tail] = APS_UT(x_array[high_tail])
            
        #Create cdf function
        inverse_cdf = interpolate.interp1d(cdf_array, x_array, kind='linear')
        
        #Generate random sample
        rnd_uniform = np.random.uniform(0, 1, N)
        rnd_sgaussian = inverse_cdf(rnd_uniform)
        
        #Find the true mode (and CI) of the skew normal
        mode, upper, lower = calculate_best(rnd_sgaussian)
        x_prec = np.arange(mode-0.01, mode+0.01, 0.00001)
        true_mode = find_mode_cdf(x_prec, lambd_best)
        
        #Rederive the upper and lower uncertainties
        mode, upper, lower = calculate_best(rnd_sgaussian, mode=true_mode)
        
        #Rescale the distribution to match input
        rnd_sgaussian -= true_mode
        scale = (sigma_plus/upper + sigma_minus/lower)/2
        rnd_sgaussian *= scale
        rnd_sgaussian += mu
        
        return rnd_sgaussian
    

def calculate_best(samples, confidence_interval=0.682689492, mode=None):
    """
    Given an array of samples, provides the mode, positive error and
    negative error.
    
    Parameters
    ---------------
        samples: array-like, dtype=float
            Mode of distribution function
        confidence_interval: float
            Confidence interval defining the upper and lower errors
        mode: float (optional)
            Mode, if known
            
    Return
    ---------------
        mode: float
            Mode of the distribution
        sigma_plus:
            Positive error
        sigma_minus:
            Negative error
    """
    n_hist, b_hist = np.histogram(samples, bins=1000)
    if mode is None:
        mode = b_hist[np.argmax(n_hist)]
    rv = st.rv_histogram((n_hist, b_hist))
    ci = rv.interval(confidence_interval)
    return mode, ci[1]-mode, mode-ci[0]

if __name__=="__main__":
    x = gaussian(4, 3, 1)

    fig = plt.figure()
    plt.hist(x, bins=1000, density=True, color='r')
    plt.yticks([])
    plt.tight_layout()
    plt.show()