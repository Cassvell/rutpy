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