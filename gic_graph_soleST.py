# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023

@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import pproc, reproc, df_dH, df_Kloc
from timeit import default_timer as timer
import sys
import pandas as pd
from get_files import get_files, list_names
import numpy as np
start = timer()

idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]
year_dir = str(idate[0:4])
df_qro = pproc('QRO', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+\
               year_dir+'/QRO/')


def df_dH_teo(date1, date2, dir_path):
    idx1 = pd.date_range(start = pd.Timestamp(date1), end =\
                         pd.Timestamp(date2+' 23:00:00'), freq='H')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                                          end = pd.Timestamp(date2), freq='D')
        
    idx_list = (idx_daylist.strftime('%Y%m%d')) 
    str1 = "teo_"
    ext = ".dst.early"
    list_fnames = list_names(idx_list, str1, ext)
    dfs_c = []
    
    for file_name in list_fnames: 
  #  for file_name in file_names:    
        df_c = pd.read_csv(dir_path+file_name, header=None, sep='\s+', \
                           skip_blank_lines=True)
        #df_c = df_c.iloc[:,4]   
        dfs_c.append(df_c) 
                
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    

    df = df.replace(999999.0, np.NaN)        
 #   idx2 = pd.date_range(start = pd.Timestamp(date1), \
  #                                    end = pd.Timestamp(date2), freq='H')
        
    df = df.set_index(idx1)
    
    df = df.loc[date1:date2]
    H  = df.iloc[:,4]

    return(H)

def df_Kloc_teo(date1, date2, dir_path):
  #  dir_path = '/home/isaac/MEGAsync/datos/Kmex/coe'

    idx1 = pd.date_range(start = pd.Timestamp(date1), end = \
                             pd.Timestamp(date2+' 21:00:00'), freq='3H')

    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                  end = pd.Timestamp(date2), freq='D')
                
    idx_list = (idx_daylist.strftime('%Y%m%d')) 
        
    str1 = "teo_"
    ext = ".index.early"
    
    list_fnames = list_names(idx_list, str1, ext)

    dfs_c = []      
    for file_name in list_fnames:    
        #print(dir_path+file_name)
        df_c = pd.read_csv(dir_path+file_name, header=None, sep='\s+').T
        df_c = df_c.iloc[:-1, :]   
        dfs_c.append(df_c) 
                    
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    
    df = df.replace(99.9, np.NaN)
          
            
   # idx2 = pd.date_range(start = pd.Timestamp(date1), \
    #                                      end = pd.Timestamp(date2), freq='3H')
            
    df = df.set_index(idx1)
        
    df = df.loc[date1:date2]
    
    k  = df.iloc[:,0]
    k = k/10
  
    return(k)

#idate = input("write initial date in format yyyy-mm-dd \n >  " )
#fdate = input("write final date in format yyyy-mm-dd \n >  " )    
gicTW_qro = (df_qro['QRO'].gic_proc[idate:fdate])
T1TW_qro = (df_qro['QRO'].T1_proc[idate:fdate])
T2TW_qro = (df_qro['QRO'].T2_proc[idate:fdate])
###############################################################################
###############################################################################


dir_path = '/home/isaac/MEGAsync/datos/dH_teo/'
H = df_dH_teo(idate, fdate, dir_path)

###############################################################################
###############################################################################
dir_path2 = '/home/isaac/MEGAsync/datos/Kmex/'
k = df_Kloc_teo(idate, fdate, dir_path2)
k = k.replace(99.9, np.NaN)
colorsValue = []
for value in k:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')

inicio = gicTW_qro.index[0]
final  = gicTW_qro.index[-1]
##############################################################################################
#fig 1
##############################################################################################
fig, ax = plt.subplots(4, figsize=(12,14))
fig.suptitle('Estudio de GICs, '+year_dir, fontsize=24, fontweight='bold')

ax[0].plot(gicTW_qro)
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_title('QRO st', fontsize=18)
ax[0].set_ylabel(' GIC [A]', fontweight='bold')

ax[1].plot(T1TW_qro, label='T1')
ax[1].plot(T2TW_qro, label='T2')
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_title('QRO st', fontsize=18)
ax[1].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[1].legend()

ax[2].plot(H, color='k')
ax[2].set_ylabel(' DH [nT]', fontweight='bold')
ax[2].set_title('Indices geomagnéticos, Estación Teoloyucan', fontsize=18)
ax[2].grid()
ax[2].set_xlim(inicio,final)

ax[3].bar(k.index, k, width = 0.1, align='edge', color=colorsValue)
ax[3].set_ylim(0,9)
ax[3].set_xlim(inicio,final)
ax[3].set_ylabel(' Kcoe', fontweight='bold')
ax[3].grid()

fig.tight_layout()

fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/2020/gic_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+"QRO.png")
plt.show()
##############################################################################################
#fig 2

plt.show()

end = timer()

print(end - start)
'''

'''
