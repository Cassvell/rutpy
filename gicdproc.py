#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 07:58:41 2021

@author: Ramon Caraballo Sept 25 2022

 funciones contenidas en este paquete:
     
pproc :  Procesamiento de datos de los sensores de CIG de ORTO en la red de 
         Potencia de México.  

reproc:  Reproceso todos los datasets de GIC

         

"""

import os
import io
import glob
#import gzip
import re
import sys
import fnmatch
import pandas as pd
import numpy as np
#import cmath
import datetime
from datetime import date
#import shutil
import ftplib
import ftputil
import fileinput
from scipy import stats
from scipy import fftpack
from scipy import signal
from ts_acc import fixer, mz_score, despike, dejump
import matplotlib.pyplot as plt    
def pproc(stid, data_dir='/home/isaac/MEGAsync/datos/gics_obs/2023/'):
    
    """ Procesamiento de datos de los sensores de CIG de ORTO en la red de 
        Potencia de México.  
    """
    os.chdir(data_dir)
    
    output = {}
    
    col_names=['Datetime','gic', 'T1','T2']
    
    if isinstance(stid, str):
        stid = stid.split()
        
        
    missing_vals = ["NaN", "NO DATA"]
        
    convert_dict = {#'Datetime': 'datetime64[ns]', #no se precisa esto
                'gic': float,
                'T1' : float,
                'T2' : float }
   
    for st in stid:
       
        fname = "gic_" + st + "*.dat"
        
        files = glob.glob(fname)
        
        # Data loading and conditioning
        
        df = pd.DataFrame([], columns=col_names)
        
        for file in files:
            
            raw = pd.read_csv(file,
                        header = None, 
                        skiprows = 1,
                        usecols = [0,1,2,3],
                        names = col_names,
                        parse_dates = [0],
                        #date_parser=(lambda col: pd.to_datetime(col, utc=True)),
                        na_values = missing_vals,
                        dtype = convert_dict,
                        low_memory= False)
                        
                
            df = pd.concat([df, raw], ignore_index=True)
            
        # Drop all axis full of NaN values
        df.dropna(axis=1, how='all', inplace=True)

        # Set index, sort and remove duplicated values keeping the first occurrence
        df.set_index('Datetime', inplace=True);
        df.sort_index(inplace = True);
        
        #Remove indexes with seconds other than '00'
        df.index = df.index.map(lambda x: x.replace(second=0))
        df = df[~df.index.duplicated(keep='first')]
        ts_start = df.index[0];
        ts_end = df.index[-1];
        
        # Reindexing & fill missing values with nan
        idx = pd.date_range(start=ts_start, end=ts_end, freq='min');
        
        df = df.reindex(idx, copy=False)
        
        # Outlier and spikes elimination
        
        for col in col_names[1:]:
            x = df[col].values
            u = fixer(x)
            
            df[col+"_proc"] = u.tolist()
        
        output.update({st:df})
        
        
    return output
        
    
#==============================================================================

def reproc(df, mod=1):
    
    """ Reproceso todos los datasets de GIC  """
    
    df['gic_dspk_cld'] = df.loc[:, 'gic_dspk']
    v = df.loc[:, 'gic_dspk_cld'].values
    
    if (mod==1):
            
        w = dejump(v, .6)
        y = v - w
        
    else:
            
            y = v - np.nanmean(v)
    
    df['gic_corr'] = y.tolist()
    
    return df

#==============================================================================
# Interpolate to fill missing values
        # dfi = df.interpolate(method='pchip', limit=60)
        # # fill bigger gaps with median of each column to ease the smoothing 
        # dfi.fillna(dfi.median(), inplace=True)
        
        # Spike removal & smoothing
        #data = {}    
    
        # for col in col_names[1:]:
        #    yy = dfi[col].values
        #    s = ts_acc(yy, 15, 7)
        #    data.update({col : s})
        
        # dfp = pd.DataFrame(data)    
        # dfp.set_index(df.index, inplace=True)    
        
        
        # dfp['gic_corr'] = signal.detrend(dfi.gic.values)
        # dfp['dT1'] = signal.detrend(dfi.T1.values)
        # dfp['dT2'] = signal.detrend(dfi.T2.values)  
        
df_qro = pproc('QRO', data_dir='/home/isaac/MEGAsync/datos/gics_obs/2023/QRO/')
df_lav = pproc('LAV', data_dir='/home/isaac/MEGAsync/datos/gics_obs/2023/LAV/')
df_rmy = pproc('RMY', data_dir='/home/isaac/MEGAsync/datos/gics_obs/2023/RMY/')
df_mzt = pproc('MZT', data_dir='/home/isaac/MEGAsync/datos/gics_obs/2023/MZT/')


#print(df_lav['LAV'].gic[0:])

idate = input("write initial date in format yyyy-mm-dd \n >  " )
fdate = input("write final date in format yyyy-mm-dd \n >  " )

    
gicTW_lav = (df_lav['LAV'].gic_proc[idate:fdate])
gicTW_qro = (df_qro['QRO'].gic_proc[idate:fdate])
gicTW_rmy = (df_rmy['RMY'].gic_proc[idate:fdate])
gicTW_mzt = (df_mzt['MZT'].gic_proc[idate:fdate])
###############################################################################
###############################################################################
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
H = dH_coe(idate, fdate)
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

quiet   = round(k/10) < 4
disturb = round(k/10) == 4
storm   = round(k/10) > 5

colorsValue = []
for value in k/10:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')



inicio = gicTW_lav.index[0]
final  = gicTW_lav.index[-1]

fig, ax = plt.subplots(5, figsize=(12,14))
fig.suptitle('Estudio de GICs, 2023', fontsize=24, fontweight='bold')

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


ax[3].plot(H, color='k')
ax[3].set_ylabel(' DH [nT]', fontweight='bold')
ax[3].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[3].grid()
ax[3].set_xlim(inicio,final)


ax[4].bar(df.index, round(k/10), width = 0.1, align='edge', color=colorsValue,\
          edgecolor= 'black')
ax[4].set_ylim(0,9)
ax[4].set_xlim(inicio,final)
ax[4].set_ylabel(' Kcoe', fontweight='bold')
ax[4].grid()

fig.tight_layout()
'''
ax[3].plot(gicTW_mzt)
ax[3].grid()
ax[3].set_xlim(inicio,final)
ax[3].set_title('MZT st', fontsize=18)
ax[3].set_ylabel(' GIC [A]', fontweight='bold')
'''