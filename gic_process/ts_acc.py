#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.stats import norm, iqr, linregress
from scipy.interpolate import interp1d

def ts_acc(y, m, wdw = 15, threshd = 7.5):

    """ This function takes a raw timeseries and remove spikes and impute
     missing values by using a combination of a Whitaker-Hayes's Algorithm
     and a Savistky-Golay filtering over the raw data.
     Args: y time series np.array
           m number of neighbours  chosen for replace spikes
           wdw window length of the savitsky golary filter, wdw > 3
    
    Subfunctions included:
        mz_score : modfied Z-score
        fixer : fix outliers in a timeseries using mz_score of the 1st diferences series
        despike : Removes spikes in a timeseries and replaces them by NaN's
        chauvenet: Outlier removal using Chauvenet criterion.
        dejump2:  Remove steps and jumps in a timeseries (simple version)
        
        
    Created on Mon Jul 27 16:26:58 2020

    @author: Ramon Caraballo """
    
    def mz_score(x):
        median_int = np.nanmedian(x)
        mad_int = np.nanmedian(np.abs(x - median_int))
        if mad_int <= 1e-15 :
            #std_dev = np.std(x)
            mean_int = np.nanmean(x)
            mean_ad_int = np.nanmean(np.abs(x - mean_int))
            mz_scores = 0.7979 * (x - median_int) / mean_ad_int 
        else:     
            mz_scores = 0.6745 * (x - median_int) / mad_int
            
        return mz_scores
 
    def fixer(y, m, threshd):
        # thereshold: binarization threshold. 
        yp = np.pad(y, (m,m+1), 'mean')
        delta = np.diff(yp, axis=0)
        spikes = np.abs(mz_score(delta)) >= threshd  #n-1
        y_out = yp.copy()                         # So we don’t overwrite y

        for i in np.arange(len(spikes)-m-1):
            if spikes[i] != 0:                    # If we have an spike in position i
                w = np.arange(i-m,i+m+1)          # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0]            # From such interval, we choose the ones which are not spikes
                #y_out[i] = np.interp(i, w2, yp[w2])  # interpolate nans within the window
                
                y_out[i] = np.nanmedian(yp[w2])   # and we take the median of their values (reemplazar por una interpolacion)
                
   

        return y_out[m:len(delta)-m]   
    
    
    # Spike removal
    s = fixer(y, m, threshd) 
    
    # Smoothing by applying a Savitsky-Golay filter: 
    sf = savgol_filter(s, wdw, 3);
    
    # # Secondary despiking to remove unwanted spikes after smoothing
    # ss = fixer(sf, m, 3.5)
    
    return sf

# Calculate modified z-scores
#==============================================================================

def mz_score(x):
    
    """ Modified z-score to identify outliers
    If MAD != 0 uses 0.6745 which is the value of the 3rd quantile in the normal
    distribution of probability.
    If MAD = 0 we approximate through the meanAD with 0.7979 the ratio between meanAD
    to the std deviation for the normal distribution

    """
    
    median_int = np.nanmedian(x)
    mad_int = np.nanmedian(np.abs(x - median_int))
    if mad_int <= 1e-15 :
        mean_int = np.nanmean(x)
        mean_ad_int = np.nanmean(np.abs(x - mean_int))
        mz_scores = 0.7979 * (x - median_int) / mean_ad_int 
    else:     
        mz_scores = 0.6745 * (x - median_int) / mad_int
        
    return mz_scores
 

# Fix & Despike a tiemseries
#==============================================================================        

def fixer(y, m=7, threshd = 7.5):
    
    """ Wittaker-Hayes  Algorithm to identify outliers in a time series """   
    # thereshold: binarization threshold. 
    
    yp = np.pad(y, (m,m+1), 'mean')
    delta = np.diff(yp, axis=0)
    spikes = np.abs(mz_score(delta)) >= threshd  #n-1
    y_out = yp.copy()                     # So we don’t o verwrite y
    for i in np.arange(len(spikes)-m-1):
        if spikes[i] != 0:               # If we have an spike in position i
            w = np.arange(i-m,i+m+1)     # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0]
            #w3 = w[spikes[w] != 0]
            # From such interval, we choose the ones which are not spikes
            y_out[i] = np.nanmedian(yp[w2])  # and we take the median of their values
            #y_out[i] = np.interp(i, w2, yp[w2])
            
    return y_out[m:len(delta)-m]   



    
def despike(y, threshd = 7.5):
    
    """ Search and replace spikes in an array with NaNs  """
       # thereshold: binarization threshold. 
       
    yp = np.pad(y, (0,1))
    delta = np.diff(yp, axis=0)
    spikes = np.abs(mz_score(delta)) >= threshd  #n
    #y_out = yp.copy()                     # So we don’t o verwrite y
    y_out = np.where(spikes !=0, np.nan, y)
    return y_out 
    
    
#==============================================================================

def chauvenet(x):
    
    """ Apply Chauvenet's criterion to remove outliers in a raw timeseries """
    
    n = np.count_nonzero(~np.isnan(x))
    P = 1 - 1/(4*n)
    
    s_mean = np.nanmean(x)
    std_dev = np.nanstd(x)
    
    dist = abs(x - s_mean)/std_dev
    # Get the quantile proportion at which is an acceptable deviation
    Dmax = abs(norm.ppf(P, loc=s_mean, scale=std_dev))
    
    xm = np.where(Dmax < dist, np.nan, x)
             
    return xm


#==============================================================================

def dejump(x, tol):
    
    """ Function to eliminate random jumps in raw data by calculating and
        substracting a constructed baseline from the original data """
    
    m = np.nanmean(x)    
    idx = np.ravel(np.asarray(np.isnan(x)).nonzero()) 
    n = len(idx)-1
    
    jumps = np.empty(len(x))
    jumps[:] = np.nan
    
    b = np.median(x[:idx[0]])
    
    jumps[:idx[0]] = b
        
    for j in np.arange(1,n+1):
        u = x[idx[j-1]:idx[j]]
        k = np.nanmedian(u)    
        jumps[idx[j-1]+1:idx[j]] = k
            
    jumps[idx[-1]+1:] = np.nanmedian(x[idx[-1]:])
        
    # p = np.pad(jumps, (0,1))
    # dj = np.diff(p)      # differentiate the jumping line, discriminate actual jumps from 
    #                      # discontinuities in the data based on a given threshold
    
    # m = np.nanmean(jumps[dj <= tol])
    
    # jumps[abs(dj) <= tol] = m   # Al false jumps are assigned the overall mean in the data

    return jumps      

#==============================================================================    

def dejump2(x, thr = 2.5 ):
    
    """ Pretends to remove steps subtracting a constructed piecewice baseline """
    # relleno los gaps nan con la mediana de los valores
    m = np.nanmedian(x)    
    idx = np.ravel(np.asarray(np.isnan(x)).nonzero())
    
    xp = x.copy()
    xp[idx] = m 
    dx = abs(np.diff(xp))
    
    #mask = np.logical_or(np.isnan(dx).nonzero(), dx >= thr)
    idx = np.ravel(np.asarray(dx >= thr).nonzero())
    idx += 1
    
    n=len(idx)-1
    
    #Descarto los spikes (escalones de ancho 1)   
    if n > 1:

        for j in range(len(idx)-1):
            if (idx[j+1]-idx[j] == 1):
                np.delete(idx, j)
                j+=1
            else:
                j+=1
            
    n = len(idx)-1
    
    jumps = np.empty(len(x))
    jumps[:] = np.nan
    
    m = np.nanmedian(x[:idx[0]])
    jumps[:idx[0]] = m
    
    
    if n < 0 :
        xout = x
    
    else:    
        for j in np.arange(1,n):
            #if (abs(np.nanmean(x[idx[j-1]:idx[j]])-np.nanmean(x[idx[j]+1:idx[j+1]])) \
             #   >= thr):
            
            u = x[idx[j-1]+1:idx[j]]
            k = np.nanmedian(u)    
            jumps[idx[j-1]+1:idx[j]] = k
            #else:
             #   u = x[idx[j-1]+1:idx[j+1]]
             #   k = np.nanmedian(u)    
             #   jumps[idx[j-1]+1:idx[j+1]] = k
                
    jumps[idx[-1]+1:] = np.nanmedian(x[idx[-1]:])
        
    xout = x-jumps
    
    return xout, jumps


# ============================================================================= 

def tsdetrend(y):
    """find linear regression line, subtract off data to detrend"""
    
    n = len(y)
    x = np.arange(0, n, 1, dtype=float)
    not_nan_ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = linregress(x[not_nan_ind],y[not_nan_ind])
    detrend_y = y - (m*x + b)
    
    return detrend_y

#==============================================================================
