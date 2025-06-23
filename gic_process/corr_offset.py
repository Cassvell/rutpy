import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data
from calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt
from gic_threshold import threshold



def corr_offset(data, threshold):
    window_size = 60  # Size of the moving window in minutes
    ndata = len(data)
    crossing_indices = []
    median_values = []
    plt.plot(data, label='GIC Data', color='blue', alpha=0.7)
    for i in range(0, ndata - window_size + 1):
        window = data[i:i + window_size]
        window_median = np.nanmedian(np.abs(window))
        median_values.append(window_median)
        # Detect threshold crossings
        if i > 0:
            prev_median = median_values[-2]
            current_median = window_median
            
            # Check for crossing (both upward and downward)
            if (prev_median <= threshold < current_median) or \
               (prev_median >= threshold > current_median):
                crossing_indices.append(i)
                #print(f'Threshold crossing at index {i} (time {i//60}h {i%60}min) '
                #      f'Median changed from {prev_median:.2f} to {current_median:.2f}')

    
    if len(crossing_indices) % 2 == 0:
        for h in range(len(crossing_indices) // 2):  # Integer division to get pair count
            start_idx = h * 2
            end_idx = h * 2 + 1
            
            if end_idx >= len(crossing_indices):
                break  # In case of odd number of indices
        
    
            sampled_data = data[crossing_indices[start_idx]:crossing_indices[end_idx]+60]
            median_w = np.nanmedian(data[crossing_indices[start_idx]:crossing_indices[end_idx]+60])
            #print(median_w)
            for i in range(len(sampled_data)):
                if sampled_data[i] > threshold or sampled_data[i] < threshold :
                    data[crossing_indices[start_idx]:crossing_indices[end_idx]+60][i] = \
                    data[crossing_indices[start_idx]:crossing_indices[end_idx]+60][i] - median_w
                
    else:
        for h in range((len(crossing_indices) // 2)+1):  # Integer division to get pair count
            if h * 2 != (len(crossing_indices) // 2)+1:           
                start_idx = h * 2
                end_idx = h * 2 + 1
                
                sampled_data = data[crossing_indices[start_idx]:crossing_indices[end_idx]+60]
                median_w = np.nanmedian(data[crossing_indices[start_idx]:crossing_indices[end_idx]+60])
                #print(median_w)
                for i in range(len(sampled_data)):
                    if sampled_data[i] > threshold or sampled_data[i] < threshold :
                        data[crossing_indices[start_idx]:crossing_indices[end_idx]+60][i] = \
                        data[crossing_indices[start_idx]:crossing_indices[end_idx]+60][i] - median_w
            else:
                start_idx = h * 2
                end_idx = data.size 
        
                sampled_data = data[crossing_indices[start_idx]:end_idx]
                median_w = np.nanmedian(data[crossing_indices[start_idx]:end_idx])
                #print(median_w)
                for i in range(len(sampled_data)):
                    if sampled_data[i] > threshold or sampled_data[i] < threshold :
                        data[crossing_indices[start_idx]:end_idx][i] = \
                        data[crossing_indices[start_idx]:end_idx][i] - median_w
        
        
        
    
    plt.plot(data, label='GIC Data offset corrected', color='black', alpha=0.7)
    plt.legend()
    plt.ylabel('GIC LAV st [A]')
    plt.show()
    return crossing_indices, median_values




i_date = sys.argv[1] 

f_date = sys.argv[2]


fyear = int(f_date[0:4])
fmonth = int(f_date[4:6])
fday = int(f_date[6:8])


finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]

idx1 = pd.date_range(start = pd.Timestamp(i_date+ ' 12:01:00'), \
                          end = pd.Timestamp(nextday + ' 12:00:00'), freq='min')


ndays = calculate_days_difference(i_date, f_date)
tot_data = (ndays+1)*1440


path = '/home/isaac/datos/gics_obs/'
file = []
stat = ['LAV', 'QRO', 'RMY', 'MZT']

gic, T1, T2 = process_station_data(i_date, f_date, path, stat[0], idx1, tot_data)

median_H = gic.resample('H').median().fillna(method='ffill')
threshold, indices = threshold(median_H, stat[0])
#print(f'Threshold for {stat[0]}: {threshold.value}')
print(f'mediana general: {np.nanmedian(gic)}')
offset = corr_offset(gic.values, threshold)

plt.plot(gic, label='GIC Data', color='blue', alpha=0.7)
plt.tight_layout()
plt.show()