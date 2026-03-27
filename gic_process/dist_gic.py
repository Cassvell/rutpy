from gicdproc import  df_gic_processed
from timeit import default_timer as timer
import sys
import pandas as pd
import os
import numpy as np
from modules.threshold import gic_threshold
from modules.window_27 import window_27

start = timer()



i_date = sys.argv[1]

# Set f_date - use provided value or default to i_date
f_date = sys.argv[2] 




stat  = ['LAV', 'QRO', 'RMY', 'MZT']
dir_path = f'/home/isaac/datos/gics_obs/processed/'
output_path = '/home/isaac/gics_rv/'

stat_dir = {}
output_file = f'{dir}'
for st in stat:
    #print(f'{st}')
    gic_data = df_gic_processed(i_date, f_date, f'{dir_path}/', st)
    gic_data = gic_data.replace(999.9, np.nan)

    stat_dir[st] = gic_data
    stat_dir[st].index = pd.to_datetime(stat_dir[st].index)    
iwindows, med_windows, fwindows, nwindows= window_27(i_date, f_date, 'date')


statistics = {}

for w in range(nwindows):

    #print(f'Window {iwindows[w]} to {fwindows[w]}')
    idx_daily = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(fwindows[w]), freq='D')
    idx_min = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(fwindows[w]), freq='min')
    
    

   
    for (idx, st)  in enumerate(stat):
        print(f'{st}')
        window_data = stat_dir[st][iwindows[w]:fwindows[w]]
        #print(window_data)
        data_array = np.array(window_data)
        
        #print(window_data)       
        non_nan_ratio = np.sum(~np.isnan(data_array)) / len(data_array)
        
        if st not in statistics:
            statistics[st] = {} 
        if non_nan_ratio > 0.2:
            n=6
            resampled_data = []
            sample = int(60/n)
            
            n_periods = int(len(data_array) // sample)
            
            for i in range(n_periods):
                start_idx = i * sample
                end_idx = (i + 1) * sample
                tmp_min = np.nanmin(data_array[start_idx:end_idx])
                tmp_max = np.nanmax(data_array[start_idx:end_idx])
                resampled_data.append([tmp_min,tmp_max])
            
            media, FWHM, valor_p = gic_threshold(resampled_data, st, w+1)
                
            statistics[st][w] = {'media': media,'FWHM': FWHM,'valor_p': valor_p}

        else:
                statistics[st][w] = {'media': np.nan,'FWHM': np.nan,'valor_p': np.nan}
                
for estacion, ventanas in statistics.items():
    df = pd.DataFrame.from_dict(ventanas, orient='index')
    df = df.reset_index().rename(columns={'index': 'ventana'})
    
    # Guardar con headers al inicio del archivo
    with open(f'{output_path}{estacion}_thresholds.minmax.csv', 'w') as f:
        # Escribir headers
        f.write('ventana,media,FWHM,valorP\n')
        
        # Escribir datos sin formato de pandas
        for _, row in df.iterrows():
            f.write(f"{int(row['ventana'])},{row['media']:.3f},{row['FWHM']:.3f},{row['valor_p']:.3f}\n")
    
    