import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data, df_gic_pp, df_gic_processed, df_sym, df_dH_exp
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
#stat = ['MZT', 'QRO', 'RMY', 'MZT']
#st = ['QRO', 'QRO', 'RMY', 'MZT']
path = f'/home/isaac/datos/gics_obs/'


finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]
idx1 = pd.date_range(start = pd.Timestamp(idate+ ' 00:00:00'), \
                          end = pd.Timestamp(fdate + ' 23:59:00'), freq='min')

ndays = calculate_days_difference(idate, fdate)
tot_data = (ndays+1)*1440


dict_gic = {'LAV': [], 'QRO': [], 'RMY': [], 'MZT': []}
pp_gic = {'LAV': [], 'QRO': [], 'RMY': [], 'MZT': []}
dict_qd = {'LAV': [], 'QRO': [], 'RMY': [], 'MZT': []}

data = process_station_data(idate, fdate, path, 'MZT', idx1, tot_data)

for i in stat:
    print(f'\n station: {i} \n')
    data = df_gic_pp(idate, fdate, path, i)
    data['gic'] = np.where((data['gic'] >= 400) | (data['gic'] <= -400), np.nan, data['gic'])
    pp_gic[i] = data['gic']
    if not data['gic'].isnull().all():

        gic_res, qd = gic_diurnalbase(data['gic'], idate, fdate, i.lower())    
        #
        dict_gic[i] = gic_res
        dict_qd[i] = qd
        
    else:
        # Create a 1-column DataFrame with idx1 as index, filled with NaN
        nan_df = pd.DataFrame({i: np.full(len(idx1), np.nan)}, index=idx1)
        dict_gic[i] = nan_df[i]  # or keep as DataFrame: dict_gic[i] = nan_df
        
        # Also create similar for dict_qd if needed
        dict_qd[i] = pd.DataFrame({i: np.full(len(idx1), np.nan)}, index=idx1)[i]
        
        
        #plt.plot(gic_res, label=f'{i} GIC no Diurnal Base', alpha=0.7)
fig, axes = plt.subplots(4, 2, figsize=(25, 20))
colors = ['blue', 'orange', 'green', 'purple']

# Processed + QD Model in left column (0, 1, 2, 3)
for i, station in enumerate(dict_gic):
    axes[i, 0].plot(pp_gic[station], label=f'{station} Processed', color=colors[i], alpha=0.7, linewidth=1.5)
    axes[i, 0].plot(idx1, dict_qd[station], label=f'{station} QD Model', color='red', alpha=0.7, linewidth=1.5)
    axes[i, 0].set_title(f'{station} - Processed GIC vs QD Model')
    axes[i, 0].set_ylabel('GIC')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

# Raw data in right column (0, 1, 2, 3)
for i, station in enumerate(dict_gic):
    axes[i, 1].plot(dict_gic[station], label=f'{station} no QD', color=colors[i], alpha=0.7, linewidth=1.5)
    axes[i, 1].set_title(f'{station} - Raw GIC Data')
    axes[i, 1].set_ylabel('GIC')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'/home/isaac/rutpy/processed/gic_processed_{idate}_{fdate}.png', dpi=300)
plt.close()
#plt.show()
#sys.exit('end of child process')
output_path = f'/home/isaac/datos/gics_obs/processed/{fyear}/'

for i in stat:
    if not os.path.exists(output_path + i):
        os.makedirs(output_path + i)

    ndays = int(len((dict_gic[i]))/1440)
    for j in range(ndays):
        start_idx = j * 1440
        end_idx = (j + 1) * 1440
        daily_data = dict_gic[i].iloc[start_idx:end_idx]
        daily_data.fillna(999.9, inplace=True)
        date_str = daily_data.index[0].strftime('%Y%m%d')
        daily_data.to_csv(output_path + f'{i}/gic_{i}_{date_str}.csv', header=False)
    #for j in 
    #dict_gic[i].to_csv(output_path + f'{i}/gic_{i}_{idate}_{fdate}.csv', header=True)


        