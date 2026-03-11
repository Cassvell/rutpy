#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023
H stations = [Coeneo, Teoloyucan, Iturbide]
@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import  df_dH, df_gic_pp
from timeit import default_timer as timer
import sys
import pandas as pd
import os.path
import os
import numpy as np
from datetime import datetime, timedelta
from modules.calc_daysdiff import calculate_days_difference
from modules.corr_offset import detect_offset
from modules.threshold import threshold
from modules.moving_window import hourly_IQR, max_IQR, med_IQR
from modules.diurnal_variation import diurnal_variation_model
start = timer()

if len(sys.argv) < 3:
    sys.exit('Usage: script.py H_start i_date [f_date]')

H_stat = sys.argv[1]
i_date = sys.argv[2]

# Set f_date - use provided value or default to i_date
f_date = sys.argv[3] if len(sys.argv) >= 4 and sys.argv[3] else i_date


    
fyear = int(f_date[0:4])
fmonth = int(f_date[4:6])
fday = int(f_date[6:8])


finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]

stat = ['QRO', 'LAV', 'RMY', 'MZT']
idx1 = pd.date_range(start = pd.Timestamp(i_date+ ' 00:00:00'),  end = pd.Timestamp(f_date + ' 23:59:00'), freq='min')
idx_daily = pd.date_range(start = pd.Timestamp(i_date),  end = pd.Timestamp(f_date ), freq='D')

ndays = calculate_days_difference(i_date, f_date)


path2 = '/home/isaac/datos/gics_obs/'
file = []

gicTW_lav = df_gic_pp(i_date, f_date, path2, stat[1])
gicTW_qro = df_gic_pp(i_date, f_date, path2, stat[0])
gicTW_rmy = df_gic_pp(i_date, f_date, path2, stat[2])
gicTW_mzt = df_gic_pp(i_date, f_date, path2, stat[3])

lav = gicTW_lav['gic']
qro = gicTW_qro['gic']
rmy = gicTW_rmy['gic']
mzt = gicTW_mzt['gic']
#T1 = gicTW_lav['T1']
#T2 = gicTW_lav['T2']
dir_path = f'/home/isaac/datos/dH_{str(H_stat)}/'

fdate = datetime.strptime(f_date, '%Y%m%d')
fdate2 = fdate + timedelta(days=1)
fdate2 = str(fdate2.strftime('%Y%m%d'))


#H = df_dH(i_date, f_date, dir_path, H_stat)
ndays = int(len(lav)/1440)

#ini_avr = np.nanmedian(lav[0:1160])
#mask  = [0, (879+(1440*(ndays-1)))]
#lav_corrected = lav.iloc[mask[0]:mask[1]] - np.nanmedian(lav.iloc[mask[0]:mask[1]])
'''
for seg_start, seg_end in correction_segments:
    segment_data = lav.iloc[seg_start:seg_end]
    corrected_segment = segment_data - np.nanmedian(segment_data)+ini_avr
    lav_corrected.iloc[seg_start:seg_end] = corrected_segment
'''

###############################################################################
###############################################################################
iqr_picks = hourly_IQR(mzt, 60, 0.7)
#iqr_picks_T1 = hourly_IQR(T1, 60, 0.7)

st = 'mzt'


threshold_gic = threshold(iqr_picks, i_date, f_date, st, '2s')
#print(iqr_picks)
#threshold_T1 = threshold(iqr_picks_T1, i_date, f_date, 'lav', '2s')

sq_gic = diurnal_variation_model(mzt, idx_daily, 10, st, 'experimental', threshold_gic, 'gic')

#sq_T1 = diurnal_variation_model(T1, idx_daily, 10, 'lav', 'experimental', threshold_T1, 'T1')
sys.exit('end HDTPM')
sq_T2 = diurnal_variation_model(T2, idx_daily, 10, 'lav', 'experimental', threshold, 'T2')


crossing_idx = detect_offset(lav, 10, 180, threshold/2)

plt.plot(lav)
plt.plot(lav[crossing_idx].index, lav[crossing_idx], 'ro', label='Cruces', markersize=8)
plt.show()





H_index = np.argmin(H)

inicio = H.index[0]
final  = H.index[-1]

# checking if the directory demo_folder  
# exist or not. 

year_dir = str(fyear) 
if not os.path.exists("/home/isaac/rutpy/gicsOutput/"+year_dir): 
      
    # if the demo_folder directory is not present  
    # then create it. 
    os.makedirs("/home/isaac/rutpy/gicsOutput/"+year_dir)     
##############################################################################################    
end = timer()

print(end - start)   

##############################################################################################
#fig 1
##############################################################################################
fig, ax = plt.subplots(2, figsize=(12,14))
fig.suptitle('Estudio de GICs, '+year_dir, fontsize=24, fontweight='bold')
ax[0].plot(gicTW_lav['gic'])
#ax[0].plot(lav[mask2[0]:mask2[1]])
#ax[0].plot(lav[mask3[0]:mask3[1]])
#ax[0].plot(lav[mask4[0]:mask4[1]])
#ax[0].plot(lav[mask5[0]:mask5[1]])
#ax[0].plot(lav[mask6[0]:mask6[1]])
#ax[0].plot(lav[mask7[0]:mask7[1]])
#ax[0].plot(lav[mask8[0]:mask8[1]])
#ax[0].plot(lav[mask9[0]:mask9[1]])
#ax[0].plot(lav[mask10[0]:mask10[1]])

#ax[0].plot(lav[mask[0]:mask[1]], color='r')
#ax[0].plot(lav_corrected, color='g')
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_title('LAV st', fontsize=18)
ax[0].set_ylabel(' GIC [A]', fontweight='bold')

ax[1].plot(H, color='k')
ax[1].set_ylabel(' DH [nT]', fontweight='bold')
ax[1].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[1].grid()
ax[1].set_xlim(inicio,final)

fig.tight_layout()

plt.show()

T1 = gicTW_lav['T1']
T2 = gicTW_lav['T2']



#mask2 = ~lav_corrected.index.duplicated(keep='first')

# Aplicar máscara a todas las series

sys.exit('end')

df_corr = pd.DataFrame({'gic':  lav_corrected, 'T1': T1, 'T2': T2})

for j in range(ndays):
    start_idx = j * 1440
    end_idx = (j + 1) * 1440
    
    if end_idx > len(df_corr):
        print(f"Skipping {j}, index out of range")
        continue

    # Slice daily data
    daily_data = df_corr[start_idx:end_idx].reset_index()     
    date = daily_data.iloc[:,0]
    tmp_year = date.iloc[0].year
    tmp_month = date.iloc[0].month
    tmp_day = date.iloc[0].day
    daily_data = daily_data.rename(columns={'index':'Datetime'})

    # Convert datetime to timestamp
    #daily_data['Datetime'] = daily_data['Datetime'].apply(
    #    lambda x: 999.9 if pd.isna(x) else x.timestamp()
    #)

    # Fill NaN values
    daily_data_filled = daily_data.fillna(999.9)
    
    output_dir = f'/home/isaac/datos/gics_obs/{tmp_year}/{stat[1]}/daily/'
    filename = f"{stat[1]}_{date.iloc[0].strftime('%Y-%m-%d')}.pp.csv"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Export using to_csv with tab separator
    daily_data_filled.to_csv(
        filepath,
        sep='\t',
        header=True,
        index=False,
        float_format='%12.7f'  # Consistent formatting
    )

print(f"Created empty files for station {stat[1]} with NaN values") 


##############################################################################################
#fig 2
##############################################################################################

