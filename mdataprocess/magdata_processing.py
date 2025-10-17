import pandas as pd
import numpy as np
#from statistics import mode
#from datetime import datetime
# Ajuste de distribuciones
import sys
#from numpy.linalg import LinAlgError
#from scipy.interpolate import splrep, splev
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter1d
#from scipy.interpolate import NearestNDInterpolator

from scipy.signal import medfilt
from lowpass_filter import aphase, dcomb
from typical_vall import night_hours, mode_nighttime, typical_value, gaus_center, mode_hourly
from threshold import get_threshold, max_IQR, med_IQR
from plots import plot_GPD, plot_detrend, plot_qdl ,plot_process
#from Ffitting import fit_data
from night_time import night_time
import os
from scipy import fftpack, signal
###############################################################################
#generación del índice temporal Date time para las series de tiempo
###############################################################################   
###############################################################################

###############################################################################}
###############################################################################
#Base line Determination
###############################################################################
###############################################################################
###############################################################################

###############################################################################
#Monthly base line
###############################################################################   
def base_line(data, net, st):    
    ndata = len(data)
###############################################################################
#Typical Day computation
###############################################################################  
    daily_mode = mode_nighttime(data, 60, net, st)
    
    
    night_data = night_hours(data, net, st)   
    

    daily_gauss = []

    for i in range(len(daily_mode)):
        tmp = night_data[i*180: ((i+1)*180)-1]
        tmp_gauss = gaus_center(tmp)
        daily_gauss.append(tmp_gauss)

    flat_daily_gauss = [item for sublist in daily_gauss for item in sublist]

    daily_sample = len(data)/1440

    daily_stacked = typical_value(daily_mode, flat_daily_gauss, daily_sample)
###############################################################################
###############################################################################
#Use of threshold for identify and Isolate disturbed days from non disturbed
###############################################################################
###############################################################################
    #We determine first an array of variation picks using Inter Quartil Range
    pickwindow = [3,4]
    original_daily_stacked = np.copy(daily_stacked)

    picks = max_IQR(data, 60, pickwindow[0], method='iqr')
    
    threshold = get_threshold(picks, st)

    # Daily IQR picks and classification
    daily_picks = med_IQR(data, 60, 24, method='iqr')

    for j in range(len(daily_stacked)):
        # Ensure daily_picks is long enough
        #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')
        if len(daily_picks) > j and ((daily_picks[j] >= threshold) or np.isnan(daily_picks[j])):
            daily_stacked[j] = np.nan
            #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')

    
    from sklearn.linear_model import LinearRegression
    
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
    
    daily_stacked_cleaned = np.array(daily_stacked)
    daily_stacked_cleaned[daily_stacked_cleaned > 29500] = np.nan 

        
    y = np.array(daily_stacked_cleaned)
    y_clean = y[~np.isnan(y)]  
    x_clean = np.arange(len(y_clean))

    x_clean = x_clean.reshape(-1, 1)  # Shape: (n_samples, n_features)

    model = LinearRegression()
    model.fit(x_clean, y_clean)

    slope = model.coef_[0]
    intercept = model.intercept_

    # Create the fit for the DAILY positions
    daily_fit = model.predict(x_clean)

    points_per_day = 1440
    daily_x_positions = np.arange(len(daily_stacked)) * points_per_day + points_per_day // 2  # Center of each day

    # Create array with NaN values preserved
    daily_fit_with_nans = np.full(len(daily_stacked_cleaned), np.nan)
    valid_indices = ~np.isnan(daily_stacked_cleaned)
    daily_fit_with_nans[valid_indices] = daily_fit

    # Get the valid positions for interpolation
    valid_x_positions = daily_x_positions[valid_indices]
    valid_y_values = daily_fit_with_nans[valid_indices]

    # Create interpolation function using ONLY valid points
    f_linear = interp1d(valid_x_positions, valid_y_values, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')

    # Interpolate for ALL daily positions
    daily_fit_interpolated = f_linear(daily_x_positions)
    #second fit
    
    x1 = np.arange(len(daily_fit_interpolated))
    y1 = daily_fit_interpolated
    
    x1 = x1.reshape(-1, 1)  # Shape: (n_samples, n_features)

    model2 = LinearRegression()
    model2.fit(x1, y1)
    
    total_days = len(daily_stacked)
    x_full_minutes = np.arange(len(data))
    x_full_days_positions = x_full_minutes / points_per_day  # Convert minutes to day units

    # Predict minute-resolution fit using the linear regression model
    full_fit = model2.predict(x_full_days_positions.reshape(-1, 1))    
        
   # plt.plot(np.arange(len(data)), data, color='k', label='Minute data')
   # plt.plot(daily_x_positions, daily_stacked_cleaned, color='b', marker='o', linestyle='', markersize=5, label='Daily picks')
    # Plot the linear fit at daily positions
   # plt.plot(daily_x_positions, daily_fit_interpolated, color='g', linewidth=2, label='Linear fit')

   # plt.plot(np.arange(len(data)), full_fit, color='m', linewidth=2, label='Linear fit min')
    
   # plt.plot(np.arange(len(data)), baseline_line, color='r', label='Baseline')
   # plt.legend()
   # plt.show()

    #plot_gpd = plot_GPD(data, picks, x, GPD, st, knee, threshold, inicio, final)
    #plot2 = plot_detrend(idate, fdate, data, original_daily_stacked,daily_stacked, st, baseline_line)
    
###############################################################################
###############################################################################
#FILL GAPS BETWEEN EMPTY DAILY VALUES    
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
    return full_fit#baseline_curve, undisturbed_days_sample

###############################################################################
#diurnal variation computation
###############################################################################
def mlt(lon, hem):
    
    
    if hem == 'W':
        lon = 360 - lon
    elif hem == 'E':
        lon = lon
        
    
    mlt = 0 + (lon / 15.0)
    if mlt < 0:
        mlt += 24
    elif mlt >= 24:
        mlt -= 24
    lt_c = 0
    
    if int(mlt) <= 12:
        lt_c = int(mlt)
    elif int(mlt) > 12:
        lt_c = int(mlt) - 24

    return lt_c


def get_diurnalvar(data, idx_daily, net, st):
    ndata = len(data)
    totdays = int(ndata/1440)
                   
    iqr_picks = max_IQR(data, 60, 24, method='stddev')    
    xaxis = np.linspace(1, 24, 1440)

#import UTC according to observatory
    ndays = 5
    info = night_time(net, st)
    
    lt = mlt(float(info[5]), info[6])
    
    utc = lt
    
    print(f"local time: {lt}")

    ini = 0
    fin = 0   
    
    try:
        utc = int(utc)  # Attempt to convert to an integer
    except ValueError:
        utc = float(utc)
    print(f"universal Coordinated time: {utc}") 

    qd_list = get_qd_dd(iqr_picks, idx_daily, 'qdl', ndays)

    qdl = [[0] * 1440 for _ in range(ndays)]

    import matplotlib.pyplot as plt
    for i in range(ndays):
        qd = (str(qd_list[i])[0:10])
        
        qd_arr = data[qd]
        
        if len(qd_arr) == 1440:
        
            qdl[i] = qd_arr


    baseline = []
###############################################################################
#diurnal variation computation
###############################################################################
    print(qd_list)

    for i in range(ndays):
        qd = (str(qd_list[i])[0:10])
        
        qd_arr = data[qd]
        
        qdl[i] = qd_arr
        
        plt.plot(xaxis, qdl[i], label=f'QD{i+1}: {qd}')
        if utc <= 0:
            ini = int(abs(utc)*60)
            fin = ini+180    
            qd_2h = qdl[i].iloc[ini:fin]    
            
            baseline_value = np.nanmedian(qd_2h)
            baseline.append(baseline_value)
        elif utc >= 0:
            ini = int(1440 - abs(utc)*60)
            if (ini+180) <= 1440:
                fin = (ini + 180)
                qd_2h = qdl[i].iloc[ini:fin]
                baseline_value = np.nanmedian(qd_2h)
                baseline.append(baseline_value)       
            else:
                fin2 = (ini+180)-1440
                
                fin1 = ini + 59
                qd_2h = qdl[i].iloc[0:fin2]   
                qd_2h2 = qdl[i].iloc[ini:fin1]
                baseline_value1 = np.nanmedian(qd_2h)
                baseline_value2 = np.nanmedian(qd_2h2)
                baseline_value = (baseline_value1 + baseline_value2)/2
                baseline.append(baseline_value)
               
        baseline_value = np.nanmedian(qd_2h)
        baseline.append(baseline_value)
        
        qdl[i] = qdl[i] - baseline_value
        
        qdl[i] = qdl[i].reset_index()
        qdl[i] = qdl[i].drop(columns=['index'])
    plt.show()
    sys.exit('End of child process')
    # Generate the average array
# Combine all DataFrames into a single DataFrame with shape (n, 1440)
    qdl_concat = pd.concat(qdl, axis=1, ignore_index=True)
    
    # Compute the mean across rows (axis=0) to get a 1x1440 array
    qd_average = qdl_concat.median(axis=1)        

    
    

    freqs = np.array([0.0, 1.1574e-5, 2.3148e-5, 3.4722e-5,4.6296e-5, \
                          5.787e-5, 6.9444e-5])    
    
    n = len(qd_average)
    N = len(qd_average)*totdays
    
    fs = 1/60
    f = fftpack.fftfreq(n, 1.0/fs)
    f = np.around(f, decimals = 9)
    mask = np.where(f >= 0)
    f=f[mask]
    
    fcomb = dcomb(n//2,1,f,freqs) 
    qd_average = np.array(qd_average)

    if np.any(np.isnan(qd_average)):
        mask2 = ~np.isnan(qd_average)
        x_interpol = np.arange(len(qd_average))
        
        from scipy import interpolate
        if qd_average.ndim == 2:
            qd_interpolated = np.zeros_like(qd_average)
            for col in range(len(qd_average)):
                col_data = qd_average
                mask_col = ~np.isnan(col_data)
                if np.any(mask_col):
                    f = interpolate.interp1d(x_interpol[mask_col], col_data[mask_col], 
                                        kind='linear', fill_value='extrapolate')
                    qd_interpolated[:, col] = f(x_interpol)
                else:
                    qd_interpolated[:, col] = 0  # or handle all-NaN column as needed
        else:
            # 1D case (for your 1440 data points array)
            mask2 = ~np.isnan(qd_average)
            if np.any(mask2):
                f = interpolate.interp1d(x_interpol[mask2], qd_average[mask2], 
                                    kind='linear', fill_value='extrapolate')
                qd_interpolated = f(x_interpol)
            else:
                qd_interpolated = np.zeros_like(qd_average)
        
        Gw = fftpack.fft(qd_interpolated, axis=0)/np.sqrt(n)
    else:
        Gw = fftpack.fft(qd_average, axis=0)/np.sqrt(n)
    

    Gw = Gw[0:n//2]
    
    G_filt = Gw*fcomb.T 
    # Remove all zero comps in G_filt
    G = G_filt[G_filt != 0]                 # 1x7
    #G = np.matrix(G)                        
    k = np.pi/720
    
    td = np.arange(N).reshape(N,1)          # Nx1
    Td = np.kron(np.ones(7), td)            # Nx7
    
    phi = aphase(G)                                             # 1x7
    X = 2*abs(G)/np.sqrt(n)                                     # 1x7
    
    Ag = np.cos(k*np.multiply(Td, np.arange(7)) + phi)          # Nx7              
    ii = np.multiply(Ag,X)                                      # Nx7      
    suma = np.sum(ii, axis=1)                                   # Nx1  
    detrd = signal.detrend(suma)   
    T = np.median(np.c_[suma, detrd], axis=1)   
    
    template = T[0:1440]
    
    plot_qdl(xaxis, template, ndays, qdl, st, idx_daily)
    sys.exit('end of child process')
    qd_offset = np.nanmedian(baseline)

    return T, qd_offset
###############################################################################
###############################################################################
###############################################################################
#AUXILIAR FUNCTIONS 
def get_qd_dd(data, idx_daily, type_list, n):
    
    daily_var = {'Date': idx_daily, 'VarIndex': data}
    
    local_var = pd.DataFrame(data=daily_var)
    
    local_var = local_var.sort_values(by = "VarIndex", ignore_index=True)
    
    if type_list == 'qdl':
    
        local_var = local_var[0:n]['Date']   
    
    elif type_list == 'I_iqr':
    
        local_var = local_var.sort_values(by = "Date", ignore_index=True)
    
    return local_var
###############################################################################
#We call the base line derivation procedures
###############################################################################  
