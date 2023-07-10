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


from gicdproc import pproc, reproc, df_dH, df_Kloc
"""
date_name = input("write initial date in format yyyy-mm-dd \n >  " )

node = ['LAV', 'MZT', 'RMY', 'QRO']

def node_dataframe(node, date):
    dir_path = '/home/isaac/MEGAsync/datos/gics_obs/2023/'

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

"""

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
dir_path = '/home/isaac/MEGAsync/datos/dH_coe/'
H = df_dH(idate, fdate, dir_path)
###############################################################################
###############################################################################
dir_path = '/home/isaac/MEGAsync/datos/Kmex/coe'
k = df_Kloc(idate, fdate, dir_path)
quiet   = round(k) < 4
disturb = round(k) == 4
storm   = round(k) > 5

colorsValue = []
for value in k:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')



inicio = gicTW_lav.index[0]
final  = gicTW_lav.index[-1]

fig, ax = plt.subplots(6, figsize=(12,14))
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


ax[5].bar(k.index, round(k), width = 0.1, align='edge', color=colorsValue)
ax[5].set_ylim(0,9)
ax[5].set_xlim(inicio,final)
ax[5].set_ylabel(' Kcoe', fontweight='bold')
ax[5].grid()

fig.tight_layout()

#fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/gic_obs_"+str(idate1)+"_"\
 #           +str(fdate1)+".png")
plt.show()
'''

'''
