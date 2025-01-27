import numpy as np
from lmoments3 import distr
import lmoments3 as lm
import kneed as kn
from scipy.stats import genpareto, kstest #anderson

def get_threshold(picks):

    ndays = int(len(picks)/4)

    picks = np.array(picks)  

    picks = picks[~np.isnan(picks)]
    
    hist, bins = np.histogram(picks, bins=ndays*2, density=True)  
 
    GPD_paramet = distr.gpa.lmom_fit(picks)

    shape = GPD_paramet['c']
    
    threshold = GPD_paramet['loc']
    
    scale = GPD_paramet['scale']
    
    x = np.linspace(min(picks), max(picks), len(picks))    
    
    GPD =  genpareto.pdf(x, shape, loc=threshold, scale=scale)
    
    GPD = np.array(GPD)
    
    if any(v == 0.0 for v in GPD):
        GPD =  genpareto.pdf(x, shape, loc=min(bins), scale=scale)
   
    params = genpareto.fit(picks)
    D, p_value = kstest(picks, 'genpareto', args=params)
    print(f"K-S test result:\nD statistic: {D}\np-value: {p_value}")
    
# Interpretation of the p-value & TEST KS for evaluating IQR picks
    alpha = 0.05

    if p_value > alpha:
        print("Fail to reject the null hypothesis: data follows the GPD")
    else:
        print("Reject the null hypothesis: data does not follow the GPD")   
        
    kneedle = kn.KneeLocator(
        x,
        GPD,
        curve='convex',
        direction='decreasing',
        S=5,
        online=True,
        interp_method='interp1d',
    )

    knee_point = kneedle.knee #elbow_point = kneedle.elbow

    print(f'knee point: {knee_point}')

#The Knee point is then considered as threshold.

    return x, GPD, knee_point
###############################################################################
###############################################################################
#generates an array of variation picks    
import numpy as np
import sys

def med_IQR(data, tw, tw_pick, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()

    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / tw)
        hourly = []
        
        for i in range(hourly_sample):
            current_window = data[i * tw : (i + 1) * tw]
            
            if len(current_window) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
            
            if non_nan_ratio > 0.9:
                QR1_hr = np.nanquantile(current_window, 0.25)
                QR3_hr = np.nanquantile(current_window, 0.75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            
            hourly.append(iqr_hr)
        
        return hourly
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanstd(trihourly_window)  # Standard deviation of the window
        else:
            stdev = np.nan  # If the window is incomplete (less than 3 data points)
        trihourly_stdev.append(stdev)

    
    daily = []
    
    # For each day, we pick the maximum IQR or standard deviation based on tw_pick
    for i in range(int(24 / tw_pick) * ndays):        
        if method == 'iqr':
            iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.90:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.9:
                iqr_picks = np.nanmedian(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)

def max_IQR(data, tw, tw_pick, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()

    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / tw)
        hourly = []
        
        for i in range(hourly_sample):
            current_window = data[i * tw : (i + 1) * tw]
            
            if len(current_window) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
            
            if non_nan_ratio > 0.9:
                QR1_hr = np.nanquantile(current_window, 0.25)
                QR3_hr = np.nanquantile(current_window, 0.75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            
            hourly.append(iqr_hr)
        
        return hourly
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanstd(trihourly_window)  # Standard deviation of the window
        else:
            stdev = np.nan  # If the window is incomplete (less than 3 data points)
        trihourly_stdev.append(stdev)

    
    daily = []
    
    # For each day, we pick the maximum IQR or standard deviation based on tw_pick
    for i in range(int(24 / tw_pick) * ndays):        
        if method == 'iqr':
            iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.90:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.9:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)