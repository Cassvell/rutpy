import numpy as np 
import sys


def hourly_IQR(data, tw, nonan_tolerance):
    ndata = len(data)
    hourly_sample = int(ndata / tw)
    hourly = []
    
    for i in range(hourly_sample):
        current_window = data[i * tw : (i + 1) * tw]
        
        if len(current_window) == 0:
            continue  # Skip empty windows
        
        non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
        
        if non_nan_ratio > nonan_tolerance:
            QR1_hr = np.nanquantile(current_window, 0.25)
            QR3_hr = np.nanquantile(current_window, 0.75)
            iqr_hr = QR3_hr - QR1_hr
        else:
            iqr_hr = np.nan
        
        hourly.append(iqr_hr)
    
    return hourly

def max_IQR(data, tw_pick, nonan_tolerance, method):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data, 60, nonan_tolerance)
    #print(hourly)

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
            #print(iqr_mov)
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > nonan_tolerance:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > nonan_tolerance:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)


def med_IQR(data, tw_pick, nonan_tolerance, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        import sys
        sys.exit()
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data, 60, nonan_tolerance)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanmedian(trihourly_window)  # Standard deviation of the window
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
            
            if non_nan_ratio > nonan_tolerance:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > nonan_tolerance:
                iqr_picks = np.nanmedian(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)
