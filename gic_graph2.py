#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023
H stations = [Coeneo, Teoloyucan, Iturbide]
@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import pproc, reproc, df_gic, df_dH, df_Kloc, fix_offset, df_dHmex, df_dst
from timeit import default_timer as timer
import sys
import pandas as pd
import os.path
import os
import numpy as np
from datetime import datetime, timedelta
from calc_daysdiff import calculate_days_difference
from ts_acc import mz_score
#from statistics import mode
import matplotlib.ticker as ticker
start = timer()

H_stat= sys.argv[1]
i_date = sys.argv[2]# "formato(yyyymmdd)"
f_date = sys.argv[3]

year_dir = str(i_date[0:4])

#print(calculate_days_difference(i_date, f_date))
fyear = int(f_date[0:4])
fmonth = int(f_date[4:6])
fday = int(f_date[6:8])


finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]

stat = ['QRO', 'LAV', 'RMY', 'MZT']
idx1 = pd.date_range(start = pd.Timestamp(i_date+ ' 12:01:00'), \
                          end = pd.Timestamp(nextday + ' 12:00:00'), freq='T')
    
idx = pd.date_range(start = pd.Timestamp(i_date), \
                          end = pd.Timestamp(f_date+ ' 23:59:00'), freq='T')
daily_index = pd.date_range(start = pd.Timestamp(i_date), \
                          
                          end = pd.Timestamp(f_date+ ' 23:59:00'), freq='D')

    
idx2 = pd.date_range(start = pd.Timestamp(i_date), \
                          end = pd.Timestamp(f_date), freq='T')
ndays = calculate_days_difference(i_date, f_date)
tot_data = (ndays+1)*1440

daily_index = daily_index.strftime("%Y-%m-%d")
path2 = '/home/isaac/datos/gics_obs/'+year_dir+'/'

file = []
SG2 = [] 

for i in (daily_index):
    SG2 = path2+stat[1]+'/daily/GIC_'+i+'_'+stat[1]+'.dat'
    #print(SG2)
    file = os.path.isfile(SG2)
   # print(file)
print(file)
if file == True:
        df_lav = df_gic(i_date, f_date,path2+stat[1]+'/daily/', stat[1])
        gicTW_lav = df_lav['LAV'].gic
        gicTW_lav = fix_offset(gicTW_lav)
        T1TW_lav = df_lav['LAV'].T1
        T2TW_lav = df_lav['LAV'].T2
else:
    df_lav = np.full(shape=(tot_data,3), fill_value=np.nan)
    df_lav = pd.DataFrame(df_lav)
  #  print(df_lav)
    df_lav = df_lav.set_index(idx1)
    gicTW_lav = df_lav.iloc[:,0]
    T1TW_lav = df_lav.iloc[:,1]
    T2TW_lav = df_lav.iloc[:,2]
###############################################################################
for i in (daily_index):
    SG2 = path2+stat[0]+'/daily/GIC_'+i+'_'+stat[0]+'.dat'
    #print(SG2)
    file = os.path.isfile(SG2)
   # print(file)
print(file)
if file == True:
        df_qro = df_gic(i_date, f_date,path2+stat[0]+'/daily/', stat[0])
        gicTW_qro = df_qro['QRO'].gic/10
        T1TW_qro = df_qro['QRO'].T1
        T2TW_qro = df_qro['QRO'].T2
else:
    df_qro = np.full(shape=(tot_data,3), fill_value=np.nan)
    df_qro = pd.DataFrame(df_qro)
    df_qro = df_qro.set_index(idx1)
    gicTW_qro = df_qro.iloc[:,0]
    T1TW_qro = df_qro.iloc[:,1]
    T2TW_qro = df_qro.iloc[:,2]
###############################################################################
for i in (daily_index):
    SG2 = path2+stat[3]+'/daily/GIC_'+i+'_'+stat[3]+'.dat'
   # print(SG2)
    file = os.path.isfile(SG2)
    #print(file)
print(file)
if file == True:
        df_mzt = df_gic(i_date, f_date,path2+stat[3]+'/daily/', stat[3])
        gicTW_mzt = df_mzt['MZT'].gic
        T1TW_mzt = df_mzt['MZT'].T1
        T2TW_mzt = df_mzt['MZT'].T2
else:
    df_mzt = np.full(shape=(tot_data,3), fill_value=np.nan)
    df_mzt = pd.DataFrame(df_mzt)
    df_mzt = df_mzt.set_index(idx1)
    gicTW_mzt = df_mzt.iloc[:,0]
    gicTW_mzt = fix_offset(gicTW_mzt)
    T1TW_mzt = df_mzt.iloc[:,1]
    T2TW_mzt = df_mzt.iloc[:,2]
###############################################################################    
for i in (daily_index):
    SG2 = path2+stat[2]+'/daily/GIC_'+i+'_'+stat[2]+'.dat'
    #print(SG2)
    file = os.path.isfile(SG2)
    #print(file)
print(file)

if file == True:
        df_rmy = df_gic(i_date, f_date,path2+stat[2]+'/daily/', stat[2])
        gicTW_rmy = df_rmy['RMY'].gic
        T1TW_rmy = df_rmy['RMY'].T1
        T2TW_rmy = df_rmy['RMY'].T2
else:
    df_rmy = np.full(shape=(tot_data,3), fill_value=np.nan)
    df_rmy = pd.DataFrame(df_rmy)
    df_rmy = df_rmy.set_index(idx1)
    gicTW_rmy = df_rmy.iloc[:,0]
    T1TW_rmy = df_rmy.iloc[:,1]
    T2TW_rmy = df_rmy.iloc[:,2]
###############################################################################
#detection of changing points

#mz_score = mz_score(gicTW_lav)
#plt.plot(gicTW_lav)


yp = np.pad(gicTW_lav, (0,1))
delta = np.diff(yp, axis=0)
spikes = np.abs(mz_score(delta)) >= 4 #n


#gicTW_lav = gicTW_lav.reset_index()
#idx = gicTW_lav[spikes].index
#gicTW_lav[spikes] = np.nan



    
'''   
for i in range(len(gicTW_lav)):
    if gicTW_lav[i] > 4.8:
        gicTW_lav[i] = np.nan    
for i in range(len(gicTW_lav)):
    if gicTW_lav[i] < -3:
        gicTW_lav[i] = np.nan
        
for j, i in enumerate(idx):
    if j % 2 == 0:  # Even increment
        gicTW_lav['gic'][idx[i:i+1]] = fix_offset(gicTW_lav['gic'][idx[i:i+1]])
    else:  # Odd increment
        gicTW_lav['gic'][idx[i+1:i+2]] = fix_offset(gicTW_lav['gic'][idx[i+1:i+2]])
  
    plt.plot(gicTW_lav['gic'][i:i+1])
'''

#print(np.mean(gicTW_lav['gic'][1069:1080]))
#print(np.mean(gicTW_lav['gic'][2097:3146]))
#print(np.mean(gicTW_lav['gic'][5298:5972]))

#gicTW_lav[3682:5078] = fix_offset(gicTW_lav[3682:5078])

#plt.plot(gicTW_lav)
#plt.plot(gicTW_lav['gic'])


###############################################################################

dir_path = '/home/isaac/datos/dH_'+str(H_stat)+'/'
H = df_dHmex('2024-05-10', f_date, dir_path, H_stat)


path_dst = '/home/isaac/datos/dst/daily/'
dst = df_dst('20240510', f_date, path_dst)
###############################################################################
path_ip = '/home/isaac/datos/ip/'
ip = pd.read_csv(path_ip+'2024-05-10.dat', header = None, sep = '\s+')
hourly_idx = pd.date_range(start = pd.Timestamp('20240510 00:00:00'), end =\
                     pd.Timestamp('20240513 23:55:00'), freq='5T')
ip = ip.set_index(hourly_idx)

ip = ip.replace(9999.99, np.nan)
B = ip.iloc[:,4]
Bx = ip.iloc[:,5]
By = ip.iloc[:,6]
Bz = ip.iloc[:,7]
###############################################################################
'''
dir_path = '/mnt/compartido/datos/Kmex/'
k = df_Kloc('2024-05-10', f_date, dir_path)
k = round(k)

colorsValue = []
for value in k:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')
'''

#modificar dependiendo de la disponibilidad de estaciones
inicio = '20240510 00:00:00'
final = f_date+' 23:00:00'
'''
if not gicTW_lav.isna().all().all():
    inicio = gicTW_lav.index[0]
    final  = gicTW_lav.index[-1]
elif not gicTW_qro.isna().all().all():
    inicio = gicTW_qro.index[0]
    final  = gicTW_qro.index[-1]
elif not gicTW_mzt.isna().all().all():
    inicio = gicTW_mzt.index[0]
    final  = gicTW_mzt.index[-1]  
''' 
# checking if the directory demo_folder  
# exist or not. 
if not os.path.exists("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir): 
      
    # if the demo_folder directory is not present  
    # then create it. 
    os.makedirs("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir)     
##############################################################################################    
end = timer()
import matplotlib.dates as mdates
# Date and time formatting
# Date and time formatting
myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
midnight_fmt = mdates.DateFormatter('%Y-%m-%d')
noon_fmt = mdates.DateFormatter('%H:%M')

import shelve
import dbm.dumb 

dir_path2 = "/home/isaac/geomstorm/rutpy/"
#with shelve.open('gicdata2024.dat', flag='r') as shelf:
#    gicdata2024 = shelf['gicdata2024']

#qro = pd.read_csv('gic_QRO_2024.csv')
#qro = qro.set_index(qro['Datetime'])
#gic_qro = qro['gic']
#T1 = qro['T1']
#T2 = qro['T2']



#gic_qro = gic_qro['2024-05-10 00:00:00': '2024-05-12 23:00:00']

#T1 = T1['2024-05-10 00:00:00': '2024-05-12 23:00:00']

#T1[T1 > 50.0] = np.nan

#T2 = T2['2024-05-10 00:00:00': '2024-05-12 23:00:00']
def custom_date_formatter(x, pos):
    dt = mdates.num2date(x)
    if dt.hour == 0:
        return dt.strftime('%Y-%m-%d')
    elif dt.hour == 12:
        return dt.strftime('%H:%M')
    else:
        return ''
##############################################################################################

##############################################################################################
fig, ax = plt.subplots(4, figsize=(12,14),  dpi=300)
#fig.suptitle('GIC detection, '+year_dir, fontsize=24, fontweight='bold')
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

y_lines = [gicTW_lav.index[1439], gicTW_lav.index[2159],
           gicTW_lav.index[2879], gicTW_lav.index[3599], gicTW_lav.index[4319]]

ax[0].plot(gicTW_lav.index[719:4980], gicTW_lav[inicio:final], color = 'k')
#ax[0].axvline(x=gicTW_lav.index[1440], color='k')
ax[0].set_xlim(gicTW_lav.index[719], gicTW_lav.index[5040])
for line in y_lines:
    ax[0].axvline(x=line, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
ax[0].grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)

ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(custom_date_formatter))
ax[0].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax[0].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax[0].xaxis.set_ticks_position('top')
ax[0].xaxis.set_label_position('top')
ax[0].text(0.05, 0.17, '(a)', transform=ax[0].transAxes, fontsize=16,
           fontweight='bold', verticalalignment='top')
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth=0.3)
ax[0].text(0.9, 0.9, 'LAV st', transform=ax[0].transAxes, fontsize=16,
           verticalalignment='top', bbox=props)
ax[0].set_ylabel(r'$GIC$ [A]')
ax[0].legend()


ax[1].plot(gicTW_rmy.index[719:4980], gicTW_rmy[inicio:final], color = 'k')
ax[1].set_xlim(gicTW_rmy.index[719], gicTW_rmy.index[4980])
ax[1].grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)
for line in y_lines:
    ax[1].axvline(x=line, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
ax[1].set_xticklabels([])
ax[1].tick_params(tick1On=False)
ax[1].text(0.05, 0.2, '(c)', transform=ax[1].transAxes, fontsize=16,
           fontweight='bold', verticalalignment='top')
ax[1].text(0.9, 0.9, 'RMY st', transform=ax[1].transAxes, fontsize=16,
           verticalalignment='top', bbox=props)
ax[1].set_ylabel(r'$GIC$ [A]')
ax[1].legend()

#ax3 = ax[3].twinx()
#ax3.plot(T1TW_mzt.index[719:6479], T1TW_mzt[inicio:final], label='T1',alpha=0.5)
#ax3.plot(T2TW_mzt.index[719:6479], T2TW_mzt[inicio:final], label='T2',alpha=0.5)
#ax3.tick_params(axis='y')
#ax3.legend()

#ax[3].plot(gicTW_mzt.index[719:6479], gicTW_mzt[inicio:final], color = 'k', \
#           label='GIC')

#y_lines = [gicTW_lav.index[1799], gicTW_lav.index[2519],
#           gicTW_lav.index[3239], gicTW_lav.index[3959], gicTW_lav.index[4679],
#           gicTW_lav.index[5399], gicTW_lav.index[6119], ]

ax[2].plot(gicTW_mzt.index[719:4980], gicTW_mzt['2024-05-10 07:00:00':'2024-05-13 06:00:00'], color = 'k')
ax[2].set_xlim(gicTW_mzt.index[719], gicTW_mzt.index[4980])
ax[2].set_xlim(gicTW_mzt.index[719], gicTW_mzt.index[4980])
ax[2].grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)
for line in y_lines:
    ax[2].axvline(x=line, color='gray', linestyle='--', linewidth=1, alpha=0.5)
#ax[3].xaxis.set_major_locator(ticker.NullLocator())
ax[2].autoscale(enable=True, axis='both', tight=True)
ax[2].set_xticklabels([])
ax[2].tick_params(tick1On=False)
ax[2].text(0.05, 0.2, '(d)', transform=ax[2].transAxes, fontsize=16,
           fontweight='bold', verticalalignment='top')
ax[2].text(0.9, 0.9, 'MZT st', transform=ax[2].transAxes, fontsize=16,
           verticalalignment='top', bbox=props)
ax[2].set_ylabel(r'$GIC$ [A]')
ax[2].legend()



vertical_lines = [dst.index[17], dst.index[20], dst.index[23], dst.index[26],
                  dst.index[28], dst.index[32], dst.index[35]]

vertical_ticks = [dst.index[int(24*0.5)], dst.index[24], dst.index[int(24*1.5)],
                  dst.index[int(24*2)], dst.index[int(24*2.5)]]

ax[3].plot(H.index, H, color='k', label=r'$\Delta H_{MEX}$')
ax[3].plot(dst.index, dst, color='r',  label='Dst')
ax[3].set_xlim(dst.index[0], dst.index[-1])

for line in vertical_ticks:
    ax[3].axvline(x=line, color='gray', linestyle='--', linewidth=1, alpha=0.5)

for line in vertical_lines:
    ax[3].axvline(x=line, color='white')

ylim = ax[3].get_ylim()

    
ax[3].grid(axis='y', linestyle='--', linewidth=1, alpha=0.5)
ax[3].set_ylabel('Geomagnetic index [nT]')
#ax4].text(0.9, 0.2, 'COE st', transform=ax[4].transAxes, fontsize=14,
#           verticalalignment='top', bbox=props)
ax[3].text(0.05, 0.2, '(f)', transform=ax[3].transAxes, fontsize=16,
           fontweight='bold', verticalalignment='top')
ax[3].xaxis.set_major_formatter(ticker.FuncFormatter(custom_date_formatter))
ax[3].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax[3].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax[3].legend(fontsize=15, loc='lower right')


fig.tight_layout()

fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir+"/gic_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+"V2.png")
plt.show()
##############################################################################################
#fig 2
##############################################################################################


