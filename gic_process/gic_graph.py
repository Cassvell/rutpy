#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023
H stations = [Coeneo, Teoloyucan, Iturbide]
@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import pproc, reproc, df_gic, df_dH, df_Kloc, fix_offset, process_station_data
from timeit import default_timer as timer
import sys
import pandas as pd
import os.path
import os
import numpy as np
from datetime import datetime, timedelta
from calc_daysdiff import calculate_days_difference
from ts_acc import mz_score


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
idx1 = pd.date_range(start = pd.Timestamp(i_date+ ' 00:00:00'), \
                          end = pd.Timestamp(f_date + ' 23:59:00'), freq='min')

ndays = calculate_days_difference(i_date, f_date)
tot_data = (ndays+1)*1440

path2 = '/home/isaac/datos/gics_obs/'
file = []

gicTW_lav, T1TW_lav, T2TW_lav = process_station_data(i_date, f_date, path2, stat[1], idx1, tot_data)

gicTW_qro, T1TW_qro, T2TW_qro = process_station_data(i_date, f_date, path2, stat[0], idx1, tot_data)
gicTW_mzt, T1TW_mzt, T2TW_mzt = process_station_data(i_date, f_date, path2, stat[3], idx1, tot_data)

gicTW_rmy, T1TW_rmy, T2TW_rmy = process_station_data(i_date, f_date, path2, stat[2], idx1, tot_data)

###############################################################################

#detection of changing points

mz_score = mz_score(gicTW_lav)
#plt.plot(gicTW_lav)

#sys.exit('end of test')

#yp = np.pad(gicTW_lav, (0,1))
#delta = np.diff(yp, axis=0)
#spikes = np.abs(mz_score(delta)) >= 10 #n


#gicTW_lav = gicTW_lav.reset_index()
#idx = gicTW_lav[spikes].index

#for j, i in enumerate(idx):
  #  if j % 2 == 0:  # Even increment
   #     gicTW_lav['gic'][idx[i:i+1]] = fix_offset(gicTW_lav['gic'][idx[i:i+1]])
    #else:  # Odd increment
     #   gicTW_lav['gic'][idx[i+1:i+2]] = fix_offset(gicTW_lav['gic'][idx[i+1:i+2]])
  
    #plt.plot(gicTW_lav['gic'][i:i+1])

#print(np.mean(gicTW_lav['gic'][1280:1914]))
#print(np.mean(gicTW_lav['gic'][1069:1080]))
#print(np.mean(gicTW_lav['gic'][2097:3146]))
#print(np.mean(gicTW_lav['gic'][5298:5972]))

#gicTW_lav['gic'][7224:7228] = fix_offset(gicTW_lav['gic'][7224:7228])

#plt.plot(gicTW_lav['gic'][2095:3146])
#plt.plot(gicTW_lav['gic'])


###############################################################################

dir_path = '/home/isaac/datos/dH_'+str(H_stat)+'/'

fdate = datetime.strptime(f_date, '%Y%m%d')
fdate2 = fdate + timedelta(days=1)
fdate2 = str(fdate2.strftime('%Y%m%d'))

H = df_dH(i_date, f_date, dir_path, H_stat)

###############################################################################
###############################################################################
dir_path = '/home/isaac/datos/Kmex/'
k = df_Kloc(i_date, f_date, dir_path, H_stat)
k = round(k)

colorsValue = []
for i, value in enumerate(k):
    if value > 9:
        k[i] = np.nan  # Set value to NaN if it's greater than 9
        colorsValue.append('gray')  # Optional: You can assign a color for NaN values
    elif value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')

k_index = np.argmax(k)
H_index = np.argmin(H)
print(f'data from: {H_stat.upper()} \n on {k.index[k_index]}, \
    max Kmex value: {np.nanmax(k)}, \n on {H.index[H_index]} min dH: {np.nanmin(H)}')

# Initialize inicio and final in case no dataset has valid data
inicio, final = None, None

# Check if any dataset contains valid data
for dataset in [gicTW_lav, gicTW_qro, gicTW_mzt, gicTW_rmy]:
    # Ensure the dataset is not empty and has a valid datetime index
    if not dataset.empty and isinstance(dataset.index, pd.DatetimeIndex):
        # Check if dataset has non-NaN values
        if not dataset.isna().all().all():
            inicio = dataset.index[0]
            final = dataset.index[-1]
            break

# Check if 'inicio' and 'final' were set
if inicio is None or final is None:
    raise ValueError("No valid dataset found, 'inicio' and 'final' could not be set.")

k = k.replace(999, np.nan)
# Check if inicio and final were defined
print(k)
print(f'max Kmex index for {H_stat}: {np.max(k)}')




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
fig, ax = plt.subplots(6, figsize=(12,14))
fig.suptitle('Estudio de GICs, '+year_dir, fontsize=24, fontweight='bold')

ax[0].plot(gicTW_lav)
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_title('LAV st', fontsize=18)
ax[0].set_ylabel(' GIC [A]', fontweight='bold')

ax[1].plot(gicTW_qro)
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_title('QRO st', fontsize=18)
ax[1].set_ylabel(' GIC [A]', fontweight='bold')

ax[2].plot(gicTW_rmy)
ax[2].grid()
ax[2].set_xlim(inicio,final)
ax[2].set_title('RMY st', fontsize=18)
ax[2].set_ylabel(' GIC [A]', fontweight='bold')

ax[3].plot(gicTW_mzt)
ax[3].grid()
ax[3].set_xlim(inicio,final)
ax[3].set_title('MZT st', fontsize=18)
ax[3].set_ylabel(' GIC [A]', fontweight='bold')

ax[4].plot(H, color='k')
ax[4].set_ylabel(' DH [nT]', fontweight='bold')
ax[4].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[4].grid()
ax[4].set_xlim(inicio,final)


ax[5].bar(k.index, k, width = 0.1, align='edge', color=colorsValue)
ax[5].set_ylim(0,9)
ax[5].set_xlim(inicio,final)
ax[5].set_ylabel(' Kcoe', fontweight='bold')
ax[5].grid()

fig.tight_layout()

fig.savefig("/home/isaac/rutpy/gicsOutput/"+year_dir+"/gic_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()
##############################################################################################
#fig 2
##############################################################################################

fig, ax = plt.subplots(6, figsize=(12,14))
fig.suptitle('Estudio de GICs, '+year_dir, fontsize=24, fontweight='bold')

ax[0].plot(T1TW_lav, label='T1')
ax[0].plot(T2TW_lav, label='T2')
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_title('LAV st', fontsize=18)
ax[0].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[0].legend()

ax[1].plot(T1TW_qro, label='T1')
ax[1].plot(T2TW_qro, label='T2')
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_title('QRO st', fontsize=18)
ax[1].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[1].legend()

ax[2].plot(T1TW_rmy, label='T1')
ax[2].plot(T2TW_rmy, label='T2')
ax[2].grid()
ax[2].set_xlim(inicio,final)
ax[2].set_title('RMY st', fontsize=18)
ax[2].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[2].legend()

ax[3].plot(T1TW_mzt, label='T1')
ax[3].plot(T2TW_mzt, label='T2')
ax[3].grid()
ax[3].set_xlim(inicio,final)

ax[3].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[3].legend()

ax[3].set_title('MZT st', fontsize=18)

ax[4].plot(H, color='k')
ax[4].set_ylabel(' DH [nT]', fontweight='bold')
ax[4].set_title('Indices geomagnéticos, Estación Teoloyucan', fontsize=18)
ax[4].grid()
ax[4].set_xlim(inicio,final)


ax[5].bar(k.index, k, width = 0.1, align='edge', color=colorsValue)
ax[5].set_ylim(0,9)
ax[5].set_xlim(inicio,final)
ax[5].set_ylabel(' Kcoe', fontweight='bold')
ax[5].grid()

fig.tight_layout()

fig.savefig("/home/isaac/rutpy/gicsOutput/"+year_dir+"/T_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()