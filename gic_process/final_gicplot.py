import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import sys
from gicdproc import df_gic_processed
from modules.window_27 import window_27
from modules.moving_window import hourly_IQR
from modules.threshold import threshold
from modules.smooth_trend import spl_fit
from modules.diurnal_variation import compute_weight
from datetime import datetime
    
    
    
idate = sys.argv[1]
fdate = sys.argv[2]

dir_path = '/home/isaac/gics_rv/'
data_path = f'/home/isaac/datos/gics_obs/processed/'

stations = ['LAV', 'QRO', 'RMY', 'MZT']
iwindows, med_windows, fwindows, nwindows= window_27('20230101', '20260207', 'date')

initial_date = datetime.strptime(idate, '%Y%m%d')
final_date = datetime.strptime(fdate, '%Y%m%d')

num_window = None
for w in range(nwindows):
    if initial_date >= iwindows[w] and final_date <= fwindows[w]:
        num_window = w
        
        #print(w)
        #break
    

    elif initial_date >= iwindows[w] and final_date > fwindows[w]:
        if w + 1 < nwindows:
            num_window = [w,w+1]
            #print(w)
            #break    

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for idx, st in enumerate(stations):
    df_stats = pd.read_csv(f'{dir_path}{st}_thresholds.minmax.csv', header=0, sep=',')
    media = df_stats.iloc[:,1]
    FWHM = df_stats.iloc[:,2]
    acc95 = df_stats.iloc[:,3]    

    if isinstance(num_window, (int, list)):
        if isinstance(num_window, int):
            media_window = media[num_window]
            threshold_window = FWHM[num_window]
        elif isinstance(num_window, list):
            media_window = (media[num_window[0]]+media[num_window[1]])/2
            threshold_window = (FWHM[num_window[0]]+FWHM[num_window[1]])/2
    
    gic_data = df_gic_processed(idate, fdate, f'{data_path}/', st)
    gic_data = gic_data.replace(999.9, np.nan)
    #print(gic_data[0:1*1440*27])       
    
    n = 3   
    fhour  = 24-n
    idx_h = pd.date_range(start = pd.Timestamp(idate),  end = pd.Timestamp(fdate + f' {fhour}:00:00'), freq=f'{n}h')
    idx_m = pd.date_range(start = pd.Timestamp(idate),  end = pd.Timestamp(fdate + f' 23:59:00'), freq='min')
    
    x_data = np.arange(0,len(idx_h))
    new_x_data = np.linspace(0, len(idx_h), len(idx_m))   
       
    sample = 60*n
    n_periods = len(gic_data) // sample
    data_sampled = []
    moving_iqr = []
    data_sampled_cleaned = []
            
        #print(window_data)       
    non_nan_ratio = np.sum(~np.isnan(np.array(gic_data))) / len(gic_data)
    if non_nan_ratio > 0.2:
        picks = hourly_IQR(gic_data, 30, 0.7)
        threshold_value= threshold(picks, idate, fdate, st, '2s')
        weights = []
        for i in range(n_periods):
            start_idx = i * sample
            end_idx = (i + 1) * sample
            tmp = np.nanmedian(gic_data[start_idx:end_idx])
            
            QR1_hr = np.nanquantile(gic_data[start_idx:end_idx], 0.25)
            QR3_hr = np.nanquantile(gic_data[start_idx:end_idx], 0.75)
            iqr_hr = QR3_hr - QR1_hr
            
            data_sampled.append(tmp)                
            if iqr_hr <= threshold_value/2:  # O la condición que necesites
                data_sampled_cleaned.append(tmp)
                moving_iqr.append(iqr_hr)
            else: 
                data_sampled_cleaned.append(np.nan)
                moving_iqr.append(np.nan)
        
        weights = compute_weight(moving_iqr, threshold_value)
        array_sampled = np.array(data_sampled_cleaned)
        
        if np.isnan(array_sampled[0]):
            array_sampled[0] = media[0]
        
        # Reemplazar último valor si es NaN
        if np.isnan(array_sampled[-1]):
            array_sampled[-1] = media[0]
        
        w = np.array(weights)

        data_1min = spl_fit(x_data, array_sampled, new_x_data, w)
    
    else:
        array_sampled = np.full(n_periods, np.nan)
        data_1min = np.full(len(idx_m), np.nan)

    threshold_up = data_1min+threshold_window#FWHM[0]/2 + media[0]
    threshold_down = data_1min-threshold_window# media[0] - FWHM[0]/2
    
    
    threshold_up_h = data_1min+threshold_window/2
    threshold_down_h = data_1min-threshold_window/2
    if st == 'QRO':
        f = 0.14
    elif st == 'LAV':
        f = 0.18
    elif st == 'RMY':
        f = 1.424
    else:
        f = 1
    
    
    axes[idx].plot(idx_m, np.array(gic_data)*f, 'b-', alpha=0.7)
    axes[idx].plot(idx_m, data_1min*f, 'k-')
    
    axes[idx].plot(idx_m, threshold_up*f, color='red',linestyle='--', linewidth=2)
    axes[idx].plot(idx_m, threshold_down*f, color='red',linestyle='--', linewidth=2)
    
    axes[idx].plot(idx_m, threshold_up_h*f, color='red',linestyle='--', linewidth=2, alpha=0.5)
    axes[idx].plot(idx_m, threshold_down_h*f, color='red',linestyle='--', linewidth=2, alpha=0.5)    

    axes[idx].grid(True, alpha=0.3, which='major')
    axes[idx].set_xlim(idx_m[0], idx_m[-601])
    #axes[idx].set_ylim(-2.5, 2.5)
    axes[idx].set_ylabel(f'{st} - GIC [A]', fontsize=14)
    axes[idx].tick_params(axis='both', which='major', labelsize=14)
    if idx==3:    
        axes[idx].set_xlabel(f'UT', fontsize=14)
plt.show()
            
            