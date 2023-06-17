#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023

@author: isaac
"""

import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import glob, os

date_name = input("write initial date in format yyyy-mm-dd \n >  " )

node = ['LAV', 'MZT', 'RMY', 'QRO']

def node_dataframe(node, date):
    dir_path = '/home/isaac/MEGAsync/datos/gics_obs/'

    df = pd.read_csv(dir_path+'Datos_GICS_'+date+' '+node+'.csv', header=0, sep=',',\
                 skip_blank_lines=True, encoding='latin1')    
    df['DateTime']  = pd.to_datetime(df.iloc[:,0], format='%Y-%m-%d %H:%M:%S')   
    df = df.set_index(df['DateTime']) 
    df = df.drop(columns=['DateTime'])
    
    gic = df.iloc[:,0]
    
    gic = gic.replace('NO DATA', np.NaN)
    return(gic)
#gic = gic.astype(str)

#rank = np.arange(float(min(gic)), 5, float(max(gic)))

gic_lav = node_dataframe(node[0],date_name)
gic_MZT = node_dataframe(node[1],date_name)
gic_RMY = node_dataframe(node[2],date_name)


###############################################################################
###############################################################################
idate = gic_lav.index[0]
idate = str(idate)[0:19]

fdate = gic_lav.index[-1]
fdate = str(fdate)[0:19]

def dH_coe(date1, date2):
    dir_path = '/home/isaac/MEGAsync/datos/dH_coe/'
    file_names  = sorted(glob.glob(dir_path+'*.early') )
    dfs_c = []
        
    for file_name in file_names:    
        df_c = pd.read_csv(file_name, header=None, sep='\s+', skip_blank_lines=True).T
        df_c = df_c.iloc[:-1, :]   
        dfs_c.append(df_c) 
                
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    
    df = df.replace(999999.0, np.NaN)
        
    day1 = file_names[0]
    dayn = file_names[-1]
    idate1 = day1[38:46]
    fdate1 = dayn[38:46] 
    idx1 = pd.date_range(start = pd.Timestamp(idate1), \
                                      end = pd.Timestamp(fdate1+' 23:00:00'), freq='H')
      
        
    idx2 = pd.date_range(start = pd.Timestamp(date1), \
                                      end = pd.Timestamp(date2), freq='H')
        
    df = df.set_index(idx1)
    
    df = df.loc[idate:fdate]
    H  = df.iloc[:,0]

    return(H)
###############################################################################
###############################################################################

#def k_coe(date1, date2):
dir_path = '/home/isaac/MEGAsync/datos/Kmex/coe'
file_names  = sorted(glob.glob(dir_path+'*.early') )
dfs_c = []
        
for file_name in file_names:    
    df_c = pd.read_csv(file_name, header=None, sep='\s+', skip_blank_lines=True).T
    df_c = df_c.iloc[:-1, :]   
    dfs_c.append(df_c) 
                
df = pd.concat(dfs_c, axis=0, ignore_index=True)    
df = df.replace(99.9, np.NaN)
        
day1 = file_names[0]
dayn = file_names[-1]
idate1 = day1[36:44]
fdate1 = dayn[36:44] 
idx1 = pd.date_range(start = pd.Timestamp(idate1), \
                                      end = pd.Timestamp(fdate1+' 21:00:00'), freq='3H')
      
        
idx2 = pd.date_range(start = pd.Timestamp(idate), \
                                      end = pd.Timestamp(fdate), freq='3H')
        
df = df.set_index(idx1)
    
df = df.loc[idate:fdate]

k  = df.iloc[:,2]

quiet = k/10 < 4
disturb = k/10 == 4
storm   = k/10 > 5

colorsValue = []
for value in k/10:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')
plt.bar(df.index, k/10, width = 0.1, color=colorsValue, edgecolor= 'black')
#plt.bar(df.index[disturb], k[disturb]/10, color='yellow' , edgecolor= 'black')
#plt.bar(df.index[storm], k[storm]/10, color='red' , edgecolor= 'black')
    return(k)

H = dH_coe(idate, fdate)
k = k_coe(idate, fdate)
###############################################################################
###############################################################################
###############################################################################
###############################################################################

inicio = gic_lav.index[0]
final  = gic_lav.index[-1]


fig, ax = plt.subplots(3, figsize=(12,12))
fig.suptitle('Estudio de GICs, 2023', fontsize=24, fontweight='bold')

ax[0].plot(gic_lav, label=node[0])
ax[0].plot(gic_MZT, label=node[1])
ax[0].plot(gic_RMY, label=node[2])
ax[0].set_title('Mediciones directas de GICs', fontsize=18)
ax[0].set_ylabel(' GIC [A]')
ax[0].legend()
ax[0].grid()
ax[0].set_xlim(inicio,final)


ax[1].plot(H, color='k')
ax[1].set_ylabel(' DH [nT]')
ax[1].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[1].grid()
ax[1].set_xlim(inicio,final)


ax[2].bar(df.index, k/10, width = 0.1, align='edge', color=colorsValue,\
          edgecolor= 'black')
ax[2].set_ylim(0,9)
ax[2].set_xlim(inicio,final)
ax[2].set_ylabel(' Kcoe')
ax[2].grid()