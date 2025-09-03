import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data
from calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt
from gic_threshold import threshold
from gic_diurnalbase import gic_diurnalbase
from corr_offset import corr_offset
import os

idate = sys.argv[1]
fdate = sys.argv[2]

fyear = int(fdate[0:4])
fmonth = int(fdate[4:6])
fday = int(fdate[6:8])

stat = ['LAV', 'QRO', 'RMY', 'MZT']

path = f'/home/isaac/datos/gics_obs/'

finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]
idx1 = pd.date_range(start = pd.Timestamp(idate+ ' 12:01:00'), \
                          end = pd.Timestamp(nextday + ' 12:00:00'), freq='min')

ndays = calculate_days_difference(idate, fdate)
tot_data = (ndays+1)*1440

file = []

dict_gic = {'LAV': [], 'QRO': [], 'RMY': [], 'MZT': []}
for i in stat:
    gic_st, T1TW, T2TW = process_station_data(idate, fdate, path, i, idx1, tot_data)

    if not gic_st.isnull().all():
        gic_res, qd = gic_diurnalbase(gic_st, i)    
        #plt.plot(gic_res, label=f'{i} GIC no Diurnal Base', alpha=0.7)
        
        if i == 'LAV':
            gic_resample = gic_res.resample('30Min').median().fillna(method='ffill')
            threshold = threshold(gic_resample)   
            gic_corr = corr_offset(gic_res, threshold[0], 60) 

            gic_corr = corr_offset(gic_corr, threshold[0], 60) 
            gic_res = gic_corr
        
        
        dict_gic[i] = gic_res
    else:
        dict_gic[i] = gic_st
        
        
        #plt.plot(gic_res, label=f'{i} GIC no Diurnal Base', alpha=0.7)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))  # 3 rows, 1 column


ax1.plot(dict_gic['LAV'], label='LAV', color='blue', alpha=0.7)
ax1.legend()

ax2.plot(dict_gic['QRO'], label='QRO', color='orange', alpha=0.7)
ax2.legend()

ax3.plot(dict_gic['RMY'], label='RMY', color='green', alpha=0.7)
ax3.legend()

ax4.plot(dict_gic['MZT'], label='MZT', color='green', alpha=0.7)
ax4.legend()


plt.tight_layout()  # Prevents label overlapping
plt.show()

output_path = f'/home/isaac/datos/gics_obs/processed/{fyear}/'

for i in stat:
    if not os.path.exists(output_path + i):
        os.makedirs(output_path + i)

    ndays = int(len((dict_gic[i]))/1440)
    for j in range(ndays):
        start_idx = j * 1440
        end_idx = (j + 1) * 1440
        daily_data = dict_gic[i].iloc[start_idx:end_idx]
        daily_data.fillna(9999.9999, inplace=True)
        date_str = daily_data.index[0].strftime('%Y%m%d')
        daily_data.to_csv(output_path + f'{i}/gic_{i}_{date_str}.csv', header=False)
    #for j in 
    #dict_gic[i].to_csv(output_path + f'{i}/gic_{i}_{idate}_{fdate}.csv', header=True)


        