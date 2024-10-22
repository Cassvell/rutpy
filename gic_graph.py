#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023
H stations = [Coeneo, Teoloyucan, Iturbide]
@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import pproc, reproc, df_gic, df_dH, df_Kloc, fix_offset
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

if file == True:
        df_lav = df_gic(i_date, f_date,path2+stat[1]+'/daily/', stat[1])
        gicTW_lav = df_lav['LAV'].gic
        print(gicTW_lav)
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

if file == True:
        df_qro = df_gic(i_date, f_date,path2+stat[0]+'/daily/', stat[0])
        gicTW_qro = df_qro['QRO'].gic
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
    T1TW_mzt = df_mzt.iloc[:,1]
    T2TW_mzt = df_mzt.iloc[:,2]
###############################################################################    
for i in (daily_index):
    SG2 = path2+stat[2]+'/daily/GIC_'+i+'_'+stat[2]+'.dat'
    #print(SG2)
    file = os.path.isfile(SG2)
    #print(file)


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


#
yp = np.pad(gicTW_lav, (0,1))
delta = np.diff(yp, axis=0)
spikes = np.abs(mz_score(delta)) >= 10 #n


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

    
print(fdate2)
H = df_dH(i_date, fdate2, dir_path, H_stat)
###############################################################################
###############################################################################
dir_path = '/home/isaac/datos/Kmex/'
k = df_Kloc(i_date, fdate2, dir_path)
k = round(k)

colorsValue = []
for value in k:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')


#modificar dependiendo de la disp de estaciones
if not gicTW_lav.isna().all().all():
    inicio = gicTW_lav.index[0]
    final  = gicTW_lav.index[-1]
elif not gicTW_qro.isna().all().all():
    inicio = gicTW_qro.index[0]
    final  = gicTW_qro.index[-1]
elif not gicTW_mzt.isna().all().all():
    inicio = gicTW_mzt.index[0]
    final  = gicTW_mzt.index[-1]  
 
# checking if the directory demo_folder  
# exist or not. 
if not os.path.exists("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir): 
      
    # if the demo_folder directory is not present  
    # then create it. 
    os.makedirs("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir)     
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

fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir+"/gic_obs_"+\
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
ax[4].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[4].grid()
ax[4].set_xlim(inicio,final)


ax[5].bar(k.index, k, width = 0.1, align='edge', color=colorsValue)
ax[5].set_ylim(0,9)
ax[5].set_xlim(inicio,final)
ax[5].set_ylabel(' Kcoe', fontweight='bold')
ax[5].grid()

fig.tight_layout()

fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir+"/T_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()




