import pandas as pd
import numpy as np
#from statistics import mode
#from datetime import datetime
# Ajuste de distribuciones
import sys
#from numpy.linalg import LinAlgError
#from scipy.interpolate import splrep, splev
from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter1d
#from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
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
def base_line(data, net, st,threshold_method):    
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
    
    threshold = get_threshold(picks, st, method=threshold_method)
    
    # Daily IQR picks and classification
    daily_picks = med_IQR(data, 60, 24, method='iqr')
   
    for j in range(len(daily_stacked)):
        # Ensure daily_picks is long enough
        #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')
        if len(daily_picks) > j and ((daily_picks[j] >= threshold) or np.isnan(daily_picks[j])):
            daily_stacked[j] = np.nan
            #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')

    #print(daily_picks)
    
    #sys.exit('end')
    from sklearn.linear_model import LinearRegression
    
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
    
    daily_stacked_cleaned = np.array(daily_stacked)
    #daily_stacked_cleaned[daily_stacked_cleaned > 29500] = np.nan 


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


    full_fit = model2.predict(x_full_days_positions.reshape(-1, 1))    
    #import matplotlib.pyplot as   
    #plt.plot(np.arange(len(data)), data, color='k', label='Minute data')
    #plt.plot(daily_x_positions, daily_stacked_cleaned, color='b', marker='o', linestyle='', markersize=5, label='Daily picks')
    # Plot the linear fit at daily positions
    #plt.plot(daily_x_positions, daily_fit_interpolated, color='g', linewidth=2, label='Linear fit')

    #plt.plot(np.arange(len(data)), full_fit, color='m', linewidth=2, label='Linear fit min')
    
    #plt.plot(np.arange(len(data)), baseline_line, color='r', label='Baseline')
    #plt.legend()
    #plt.show()
    #sys.exit('end')
    #plot_gpd = plot_GPD(data, picks, x, GPD, st, knee, threshold, inicio, final)
    #plot2 = plot_detrend(idate, fdate, data, original_daily_stacked,daily_stacked, st, baseline_line)
 
###############################################################################
###############################################################################
#FILL GAPS BETWEEN EMPTY DAILY VALUES    
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
    return baseline_line#full_fit#baseline_curve, undisturbed_days_sample

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


def get_diurnalvar(data, idx_daily, net, st, qd_method, threshold_method):
    ndata = len(data)
    totdays = int(ndata/1440)
                   
    iqr_picks = max_IQR(data, 60, 24, method='stddev')    
    xaxis = np.linspace(0, 23, 1440)

#import UTC according to observatory

    #threshold = get_threshold(iqr_picks, st, 'GPD')
    ini = 0
    fin = 0      
    
    if qd_method == 'qd5':   
        ndays = 5
        info = night_time(net, st)
        
        lt = mlt(float(info[5]), info[6])
        
        utc = lt
        
        print(f"local time: {lt}")

 
        
        try:
            utc = int(utc)  # Attempt to convert to an integer
        except ValueError:
            utc = float(utc)
        print(f"universal Coordinated time: {utc}") 

        qd_list = get_qd_dd(iqr_picks, idx_daily, 'qdl', ndays)
    
    
    elif qd_method == 'experimental':
        qd_list = get_qd_dd(iqr_picks, idx_daily, 'I_iqr', ndays)
        threshold = get_threshold(iqr_picks, st, threshold_method)    
        exceeding_count = (qd_list.iloc[:, 1] > threshold).sum()

        if exceeding_count > 0:
            print(f"Found {exceeding_count} values exceeding threshold")
            mask = qd_list.iloc[:, 1] > threshold
            qd_list.loc[mask, qd_list.columns[1]] = np.nan
            qd_list_nonan = qd_list[~qd_list.iloc[:, 1].isna()]
        else:
            print("No values exceed threshold")
            qd_list_nonan = qd_list[~qd_list.iloc[:, 1].isna()]

            #qd_list_nonan = qd_list[~qd_list.iloc[:, 1].isna()]   
        qd_list = qd_list_nonan  

    ndays = len(qd_list)
    qdl = [[0] * 1440 for _ in range(ndays)]
    
    baseline = []
    #sys.exit('end')
###############################################################################
#diurnal variation computation
###############################################################################
    import weightedstats
    
    for i in range(ndays):
        if qd_method == 'experimental':
            qd = str(qd_list.iloc[i, 0])[0:10]
            iqr = qd_list.iloc[i, 1]     
        elif qd_method == 'qd5':
            qd = qd_list.iloc[i].strftime('%Y-%m-%d')
            
        qd_arr = data[qd]        
        #plt.plot(xaxis, qd_arr)
        if len(qd_arr) == 1440:
            qdl[i] = qd_arr
#print(data['2024-05-25'])
                   
        # Check if qdl[i] is not None before processing
        if qdl[i] is not None:
            if utc <= 0:
                ini = int(abs(utc) * 60)
                fin = ini + 180    
                
                # Proper slicing for pandas Series/DataFrame
                if hasattr(qdl[i], 'iloc'):
                    qd_2h = qdl[i].iloc[ini:fin]  # For pandas objects
                else:
                    qd_2h = qdl[i][ini:fin]  # For lists/arrays
                
                baseline_value = np.nanmedian(qd_2h)
                baseline.append(baseline_value)
                
            elif utc >= 0:
                ini = int(1440 - abs(utc) * 60)
                if (ini + 180) <= 1440:
                    fin = ini + 180
                    if hasattr(qdl[i], 'iloc'):
                        qd_2h = qdl[i].iloc[ini:fin]
                    else:
                        qd_2h = qdl[i][ini:fin]
                    baseline_value = np.nanmedian(qd_2h)
                    baseline.append(baseline_value)       
                else:
                    fin2 = (ini + 180) - 1440
                    fin1 = 1440  # Go to end of day
                    
                    if hasattr(qdl[i], 'iloc'):
                        qd_2h1 = qdl[i].iloc[0:fin2]   
                        qd_2h2 = qdl[i].iloc[ini:fin1]
                    else:
                        qd_2h1 = qdl[i][0:fin2]   
                        qd_2h2 = qdl[i][ini:fin1]
                        
                    baseline_value1 = np.nanmedian(qd_2h1)
                    baseline_value2 = np.nanmedian(qd_2h2)
                    baseline_value = (baseline_value1 + baseline_value2) / 2
                    baseline.append(baseline_value)
            
            # Remove duplicate baseline.append - you already did it above
            # baseline_value = np.nanmedian(qd_2h)
            # baseline.append(baseline_value)
            
            if len(qdl[i]) == 1440:
                # Create xaxis if not defined
                #plt.plot(xaxis, qdl[i] - baseline_value)
                qdl[i] = qdl[i] - baseline_value
                
            # Reset index if it's a pandas object
            if hasattr(qdl[i], 'reset_index'):
                qdl[i] = qdl[i].reset_index(drop=True)
            # Ensure it's a 1D array
            qdl[i] = np.array(qdl[i]).flatten()
            
            # Append iqr to create array of length 1441
            if qd_method == 'experimental':
                qdl[i] = np.append(qdl[i], iqr)
    
    
    if qd_method == 'experimental':
        if qdl:
            data_arrays = [arr[:-1] for arr in qdl]
            iqr_values = [arr[-1] for arr in qdl]
            
            qdl_df = pd.DataFrame(np.array(data_arrays).T)
            
            # Calculate weights
            weights_list = []
            for iqr in iqr_values:
                if iqr < threshold:
                    weight = 1 / (threshold * 2)
                elif iqr < (threshold / 1.5):
                    weight = 1 / (threshold * 1.5)
                elif iqr < (threshold / 2):
                    weight = 1 / threshold
                else:
                    weight = 1.0
                weights_list.append(weight)
            
            weights_array = np.array(weights_list)
            weights_array = weights_array / np.mean(weights_array)
            
            # Process 30-minute segments
            qd_30min_median_list = []
            qd_30min_std_list = []
            
            for j in range(48):  # 48 segments of 30 minutes
                start_idx = j * 30
                end_idx = (j + 1) * 30
                
                # Extract 30-minute segment for all days
                segment_df = qdl_df.iloc[start_idx:end_idx]
                
                # Calculate weighted median across days for this time segment (NO FILTER)
                segment_median = segment_df.apply(
                    lambda col: weightedstats.weighted_median(col, weights=weights_array),
                    axis=1
                ).median()  # Median of the 30 minutes
                
                # Calculate standard deviation across days for this time segment (NO FILTER)
                segment_std = segment_df.std(axis=1).median()  # Median std of the 30 minutes
                
                qd_30min_median_list.append(segment_median)
                qd_30min_std_list.append(segment_std)
            
            # Interpolate both back to 1440 points
            x_30min = np.arange(48)
            x_1440 = np.linspace(0, 47, 1440)
            
            qd_average = np.interp(x_1440, x_30min, qd_30min_median_list)
            qd_std_raw = np.interp(x_1440, x_30min, qd_30min_std_list)
            
            # Apply moving median filter ONLY to the standard deviation array
            window_size = 60
            qd_std = pd.Series(qd_std_raw).rolling(window=window_size, center=True, min_periods=1).median().values
            
        else:
            qd_average = None
            qd_std = None    
    
    elif qd_method == 'qd5':
        if qdl:
            data_arrays = [arr[:] for arr in qdl]
            #iqr_values = [arr[-1] for arr in qdl]        
            qdl_df = pd.DataFrame(np.array(data_arrays).T)
            qd_average = qdl_df.median(axis=1)
            #print(qdl_df)
            
    
    #for i in range(ndays):
    #    if len(qdl[i]) == 1440:
     #       plt.plot(xaxis, qdl[i], alpha=0.6)                                
            
    plt.plot(xaxis, qd_average, color='blue', linewidth=2)

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
    
    inicio = idx_daily[0].strftime('%Y-%m-%d')
    final = idx_daily[-1].strftime('%Y-%m-%d')    
    
    path = '/home/isaac/MEGA/posgrado/doctorado/'
    #plt.plot(xaxis, template, label="model", color='k',linewidth=4.0 )
    #plt.plot(xaxis, template + qd_std, color = 'red', linestyle='--',linewidth=3)
    #plt.plot(xaxis, template - qd_std, color = 'red', linestyle='--', linewidth=3)    
    #for i in range(ndays):
    #    if len(qdl[i]) == 1440:
    #        plt.plot(xaxis, qdl[i], alpha=0.6)
    
    #plt.xlim(0,23)
    #plt.savefig(f'{path}semestre5/qdl/{st}_{inicio}_{final}_gics.png')
    #plt.title(f'{st.upper()} diurnal variation')       
    #plt.legend()
    #plt.tight_layout() 
    #plt.show()

    
        #archivos de salida 2: 
        #modelo QDL y stddev con máximas y mínimas amplitudes
    if qd_method == 'experimental':    
        max_arg = np.argmax(template)
        min_arg = np.argmin(template)
        #, 'extemes' : [np.max(template), np.min(template), qd_std[max_arg], qd_std[min_arg]]
        output = {'model' : template, 'stddev' :  qd_std}
        df = pd.DataFrame(output).fillna(999.9)
        
        path2 = f'/home/isaac/datos/gics_obs/qdl/{st.upper()}/'
        
        directory = os.path.dirname(path2)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        
        
        doi_1 = idx_daily[0].timetuple().tm_yday
        doi_2 = idx_daily[-1].timetuple().tm_yday
        
        output_name = f'{st}_{doi_1}_{doi_2}.qdl.dat'    
        full_path = os.path.join(directory, output_name)
        with open(full_path, 'w') as f:
            for _, row in df.iterrows():
                # Format both columns as floats with F10.7 format (10 characters wide, 7 decimal places)
                model_str = f"{row['model']:10.7f}"
                stddev_str = f"{row['stddev']:10.7f}"
                # Write line with 3 spaces separation
                f.write(f"{model_str}   {stddev_str}\n")
    
    
    #sys.exit('end of child process')
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
