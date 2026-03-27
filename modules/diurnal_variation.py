import numpy as np
import pandas as pd
import weightedstats as ws
import matplotlib.pyplot as plt
import sys
import os
from modules.obs_info import obs_mlon, obs_mlt
from modules.moving_window import max_IQR, med_IQR, hourly_IQR
from modules.corr_offset import detect_offset
from scipy import fftpack, signal
from modules.lowpass_filter import aphase, dcomb
from modules.smooth_trend import fourier_fit, spl_fit, fourier_fit_with_freqs

def get_qd_dd(data1, data2, idx_daily, type_list, n):
    
    daily_var = {'Date': idx_daily, 'VarIndexMed': data1, 'VarIndexMax' : data2}
    
    local_var = pd.DataFrame(data=daily_var)
    
    local_var = local_var.sort_values(by = "VarIndexMed", ignore_index=True)
    
    if type_list == 'qdl':
    
        local_var = local_var[0:n]['Date']   
    
    elif type_list == 'I_iqr':
    
        local_var = local_var.sort_values(by = "Date", ignore_index=True)
    
    return local_var

def compute_weight(iqr_value, threshold):
        return 1.0 - (iqr_value / threshold)

def median_qdl_weighted(qdl_data, threshold):
    if qdl_data:
        data_arrays = [item['qdl'] for item in qdl_data]
        
        baselines = [item['baseline'] for item in qdl_data]
        iqr_max = [item['iqr_med'] for item in qdl_data]
        iqr_med = [item['iqr_max'] for item in qdl_data]
        
        qdl_df = pd.DataFrame(np.array(data_arrays).T)
        
        weights_med = [compute_weight(iqr_med, threshold)]
        weights_max = [compute_weight(iqr_max, threshold)]
        

        avr_weight = [(w1 + w2)/2 for w1, w2 in zip(weights_med, weights_max)]
        avr_weight_array = np.array(avr_weight).flatten()
        baselines_array = np.array(baselines)
        qd_30min_median_list = []
        qd_30min_std_list = []
    
        for j in range(48):  # 48 segments of 30 minutes
            start_idx = j * 30
            end_idx = (j + 1) * 30
            
            tmp_medians = []
            tmp_std = []
            
            for day_idx in range(len(avr_weight_array)):  # n días
                #por cada dia, se extrae cada media hora y se determina la mediana
                day_values = qdl_df.iloc[start_idx:end_idx, day_idx].values               
                median_val = np.nanmedian(day_values)
                #print(day_values)
                std_val = np.nanstd(day_values)
                tmp_medians.append(median_val)
                tmp_std.append(std_val)

            #genera un arreglo de 48 medianas por n cantidad de dias
            medians = np.array(tmp_medians)
            segment_std = np.median(tmp_std)
            
            
            if len(avr_weight_array) >= 4:
                segment_median = ws.weighted_median(medians, weights=avr_weight_array)  
            else: 
                segment_median = np.nanmedian(medians)
                
            qd_30min_median_list.append(segment_median)
            qd_30min_std_list.append(segment_std)
        
        if len(avr_weight_array) >= 4:
            time_window_baseline = ws.weighted_median(baselines_array, weights=avr_weight_array)
        else:
            time_window_baseline =  np.nanmedian(baselines_array)
        # Interpolate both back to 1440 points
        x_30min = np.arange(48)
        x_1440 = np.linspace(0, 47, 1440)
        #qd_average = np.interp(x_1440, x_30min, qd_30min_median_list)
        qd_std_raw = np.interp(x_1440, x_30min, qd_30min_std_list)
        qd_average = qd_30min_median_list
        # Apply moving median filter ONLY to the standard deviation array
        window_size = 60
        qd_std = pd.Series(qd_std_raw).rolling(window=window_size, center=True, min_periods=1).median().values
        monthly_baseline = np.full(1440, time_window_baseline)
    else:
        qd_average = np.full(48, np.nan)
        qd_std = np.full(48, np.nan)
        monthly_baseline = np.full(1440, np.nan)    
    
    return qd_average, qd_std, monthly_baseline

def lowpass_filtering(qd_average, totdays):
    x_1440 = np.linspace(0, 47, 1440)
    x_30min = np.arange(48)
    qd_average = np.interp(x_1440, x_30min, qd_average)
    
    freqs = np.array([0.0, 1.1574e-5, 2.3148e-5, 3.4722e-5,4.6296e-5, \
                          5.787e-5, 6.9444e-5])    
    
    n = len(qd_average)
    N = len(qd_average)
    
    if not np.all(np.isnan(qd_average)):
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
    else:
        T = np.full(n, np.nan)    
    return T

def diurnal_variation_model(data, idx_daily, jump_val, st, qd_method, threshold, data_type):
    ndata = len(data)
    totdays = int(ndata/1440)
    
    iqr_picks = hourly_IQR(data, 60, 0.7)    

              
    iqr_maxpicks = max_IQR(data, 24, 0.8, method='iqr')
    
    iqr_medpicks = med_IQR(data, 24, 0.8, method='stddev')    
    xaxis = np.linspace(0, 23, 1440)
    
    mlon = obs_mlon(st)
   #mlt_series = compute_mlt_ts(mlon, data.index, idx_daily)
    dt = data.index[0]
    mlt_diff = obs_mlt(mlon, dt)

    ini = 0
    fin = 0      

    if qd_method == 'qd5':   
        ndays = 5

        try:
            mlt_diff = int(mlt_diff)  # Attempt to convert to an integer
        except ValueError:
            mlt_diff = float(mlt_diff)
        print(f"universal Coordinated time: {mlt_diff}") 

        qd_list = get_qd_dd(iqr_medpicks, idx_daily, 'qdl', ndays)
    
    
    elif qd_method == 'experimental':
        qd_list = get_qd_dd(iqr_medpicks,iqr_maxpicks ,idx_daily, 'I_iqr', totdays) 
       
        #print(qd_list)
        exceeding_count1 = (qd_list.iloc[:, 1] > threshold).sum()
        exceeding_count2 = (qd_list.iloc[:, 2] > threshold).sum()
        
        
        exceeding_count = exceeding_count1 + exceeding_count2
        

        if exceeding_count > 0:
            print(f"Found {exceeding_count} values exceeding threshold")
            mask1 = qd_list.iloc[:, 1] > threshold
            mask2 = qd_list.iloc[:, 2] > threshold
            mask = mask1 | mask2
            qd_list.loc[mask, qd_list.columns[[1, 2]]] = np.nan
            
            
        else:
            print("No values exceed threshold")
        
        qd_list_nonan = qd_list[~qd_list.iloc[:, 1].isna()]

            #qd_list_nonan = qd_list[~qd_list.iloc[:, 1].isna()]   

    ndays = len(qd_list_nonan)
    qdl_data = []
    qdl = [[0] * 1440 for _ in range(ndays)]
    baseline = []

###############################################################################
#diurnal variation computation
###############################################################################

    
    for i in range(ndays):
        if qd_method == 'experimental':
            qd = str(qd_list_nonan.iloc[i, 0])[0:10]
            iqr_med = qd_list_nonan.iloc[i, 1]
            iqr_max = qd_list_nonan.iloc[i, 2]
        elif qd_method == 'qd5':
            qd = qd_list_nonan.iloc[i].strftime('%Y-%m-%d')
           
        qd_arr = data[qd] 
       
        non_nan_ratio = np.sum(~np.isnan(qd_arr)) / len(qd_arr)
        
        if non_nan_ratio > 0.9:
            qdl[i] = qd_arr
            
#print(data['2024-05-25'])
            #print(qdl[i])     
            if not np.all(np.isnan(qdl[i])):
                crossing_idx = detect_offset(qdl[i], jump_val, 120, threshold/2)

                if len(crossing_idx) == 0:
                
                    if mlt_diff <= 0:
                        ini = int(abs(mlt_diff) * 60)
                        fin = ini + 180    
                        
                        # Proper slicing for pandas Series/DataFrame
                        if hasattr(qdl[i], 'iloc'):
                            qd_nh = qdl[i].iloc[ini:fin]  # For pandas objects
                        else:
                            qd_nh = qdl[i][ini:fin]  # For lists/arrays
                        
                        baseline_value = np.nanmedian(qd_nh)
                        baseline.append(baseline_value)
                        
                    elif mlt_diff >= 0:
                        ini = int(1440 - abs(mlt_diff) * 60)
                        if (ini + 180) <= 1440:
                            fin = ini + 180
                            if hasattr(qdl[i], 'iloc'):
                                qd_nh = qdl[i].iloc[ini:fin]
                            else:
                                qd_nh = qdl[i][ini:fin]
                            baseline_value = np.nanmedian(qd_nh)
                            baseline.append(baseline_value)       
                        else:
                            fin2 = (ini + 180) - 1440
                            fin1 = 1440  # Go to end of day
                            
                            if hasattr(qdl[i], 'iloc'):
                                qd_nh1 = qdl[i].iloc[0:fin2]   
                                qd_nh2 = qdl[i].iloc[ini:fin1]
                            else:
                                qd_nh1 = qdl[i][0:fin2]   
                                qd_nh2 = qdl[i][ini:fin1]
                                
                            baseline_value1 = np.nanmedian(qd_nh1)
                            baseline_value2 = np.nanmedian(qd_nh2)
                            baseline_value = (baseline_value1 + baseline_value2) / 2
                            baseline.append(baseline_value)
    
                    qdl_corr = qdl[i] - baseline_value
                    # Create xaxis if not defined
                    plt.plot(xaxis, qdl_corr, color='gray',alpha=0.3)
                        
                    # Reset index if it's a pandas object
                    if hasattr(qdl_corr, 'reset_index'):
                        qdl_corr = qdl_corr.reset_index(drop=True)
                    # Ensure it's a 1D array
                    qdl_corr = np.array(qdl_corr).flatten()
                    
                    tmp_dict = {
                    'qdl': qdl_corr,  # Array de 1440
                    'fecha': qd,
                    'iqr_med': iqr_med if qd_method == 'experimental' else None,
                    'iqr_max': iqr_max if qd_method == 'experimental' else None,
                    'baseline': baseline_value
                }    
                    qdl_data.append(tmp_dict)
        #plt.show()
        # else:
            
    
    if qd_method == 'experimental':
        qd_average, qd_std, monthly_baseline = median_qdl_weighted(qdl_data, threshold)
    #print(qd_std)
    #T = lowpass_filtering(qd_average, totdays)
    tdata = np.arange(0, 24, 0.5) * 3600  # horas -> segundos
    new_tdata = np.linspace(0, 24, 1440) * 3600 


    freqs = np.array([0.0, 1.1574e-5, 2.3148e-5, 3.4722e-5, 4.6296e-5, 
                  5.787e-5, 6.9444e-5])
    FF = fourier_fit_with_freqs(tdata, qd_average, new_tdata, freqs)
       #Fourier Series Fitting
    
    plt.plot(np.linspace(0,23,48), qd_average, 'ko', markersize=6)
    #plt.plot(xaxis, T, color='r', linewidth=2)         
    plt.plot(xaxis, FF, color='r', linewidth=3)  
    inicio = idx_daily[0].strftime('%Y-%m-%d')
    final = idx_daily[-1].strftime('%Y-%m-%d')    
    
    path = '/home/isaac/gics_rv/fig/'
    #print(len(T))
    #plt.plot(xaxis, T, label="model", color='k',linewidth=4.0 )
    #plt.plot(xaxis, T + qd_std, color = 'red', linestyle='--',linewidth=3)
    #plt.plot(xaxis, T - qd_std, color = 'red', linestyle='--', linewidth=3)    
    

    #for i in range(ndays):
    #    if len(qdl[i]) == 1440:
    #        plt.plot(xaxis, qdl[i], alpha=0.6)
    
    plt.xlim(0,23)
    plt.title(f'{st.upper()} diurnal variation')       
    plt.ylabel('Quiet day records [A]')
    plt.xlabel('UT [h]')
    plt.tight_layout() 
    path_figure = f'{path}/QDL_GICS/{st.upper()}/'
    directory = os.path.dirname(path_figure)
    if directory and not os.path.exists(path_figure):
        os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{path_figure}{st}_{inicio}_{final}_{data_type}.png')
    plt.close()

        #archivos de salida 2: 
        #modelo QDL y stddev con máximas y mínimas amplitudes
    if qd_method == 'experimental':    
        output = {'model' : FF, 'stddev' :  qd_std, 'baseline' : monthly_baseline}
        df = pd.DataFrame(output).fillna(999.9)
        
        path2 = f'/home/isaac/datos/gics_obs/qdl/{st.upper()}/'
        
        directory = os.path.dirname(path2)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        
        
        doi_1 = idx_daily[0].timetuple().tm_yday
        doi_2 = idx_daily[-1].timetuple().tm_yday
        year_1 = idx_daily[0].strftime('%Y') 
        year_2 = idx_daily[-1].strftime('%Y') 
        
        
        output_name = f'{st}_{year_1}-{doi_1}_{year_2}-{doi_2}.qdl.{data_type}.dat'    
        full_path = os.path.join(directory, output_name)
        with open(full_path, 'w') as f:
            for _, row in df.iterrows():
                # Format both columns as floats with F10.7 format (10 characters wide, 7 decimal places)
                model_str = f"{row['model']:10.7f}"
                stddev_str = f"{row['stddev']:10.7f}"
                baseline_str = f"{row['baseline']:10.7f}"
                # Write line with 3 spaces separation
                f.write(f"{model_str},{stddev_str},{baseline_str} \n")

        print(f'file: {full_path} created')
    #qd_offset = np.nanmedian(baseline)

    return FF, monthly_baseline