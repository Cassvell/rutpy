#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:22:19 2023

@author: isaac
"""
import os

import glob
#import gzip
import pandas as pd
import numpy as np
#import cmath
#import shutil
import ftplib
import ftputil
import fileinput
from scipy import stats
from scipy import fftpack
from scipy import signal
from ts_acc import fixer, mz_score, despike, dejump
from gicdproc import pproc
import sys

year_dir = sys.argv[1]# "formato(yyyymmdd)"
stat = sys.argv[2]

data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/'+stat+'/'
list_fnames = glob.glob(data_dir + "/*.dat")
dfs_c = []

missing_vals = ["NaN", "NO DATA"]
col_names=['Datetime','gic', 'T1','T2']
convert_dict = {#'Datetime': 'datetime64[ns]', #no se precisa esto
                'gic': float,
                'T1' : float,
                'T2' : float }

        
for file_name in list_fnames: 
  #  for file_name in file_names:    
    df_c = pd.read_csv(file_name, 
                       header=None, 
                       skiprows = 1,
                       usecols = [0,1,2,3],
                       names = col_names,
                       parse_dates = [0])  
    dfs_c.append(df_c) 
df = pd.concat(dfs_c, axis=0, ignore_index=True)



df.dropna(axis=1, how='all', inplace=True)
        # Drop all axis full of NaN values


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
df = df.reset_index()  
################################################################################
################################################################################
#fragmentación del archivo original en varios archivos con un día de ventana de
#tiempo
################################################################################
################################################################################
step=1440
fhour=1439
#print(df)

for i in range(0,len(idx)-1,step):
    mask = (df['index'] >= idx[i]) & (df['index'] <= idx[i+1439])
    df_new = df[mask]
   # sym_H  = df_new['SYM-H'].apply(lambda x: '{0:0>4}'.format(x))
    date = str(idx[i])
    date = date[0:10]
    
    #df_new = df_new.drop(columns=['DOY', 'ASY-D', 'SYM-D', 'ASY-H'])
    df_new = df_new.rename(columns={'index':'Datetime'})
               
    name_new = 'GIC_'+date+'_'+stat+'.dat'
    new_path = data_dir+'/daily/'
            
    df_new.to_csv(new_path+name_new, sep= '\t', index=False)        

