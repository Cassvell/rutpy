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
from datetime import datetime, timedelta


from get_files import get_files, list_names

def pproc(stid, data_dir):
    
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



#print(df_lav['LAV'].gic[0:])
###############################################################################
###############################################################################
def df_gic(date1, date2, dir_path, stat):
    col_names = ['Datetime','gic', 'T1','T2', 'gic_proc', 'T1_proc',	'T2_proc']
    
    fyear = int(date2[0:4])
    fmonth = int(date2[4:6])
    fday = int(date2[6:8])


    finaldate= datetime(fyear, fmonth,fday)
    nextday = finaldate+timedelta(days=1)
    nextday = str(nextday)[0:10]
    
    
    idx1 = pd.date_range(start = pd.Timestamp(str(date1)+' 12:01:00' ), end =\
                         pd.Timestamp(nextday+' 12:00:00'), freq='T')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(str(date1)), \
                                          end = pd.Timestamp(str(date2)), freq='D')
        
    idx_list = (idx_daylist.strftime('%Y-%m-%d')) 
    str1 = "GIC_"
    ext = "_"+stat+".dat"

   # remote_path= '/data/output/indexes/'+station+'/'
    list_fnames = list_names(idx_list, str1, ext)
    #print(list_fnames)
   # wget = get_files(date1, date2, remote_path, dir_path, list_fnames)
    dfs_c = []
    missing_vals = ["NaN", "NO DATA"]    
    for file_name in list_fnames: 
  #  for file_name in file_names:    
        df_c = pd.read_csv(dir_path+file_name, header=None, skiprows = 1, sep='\s+', \
                           parse_dates = [0], na_values = missing_vals,)
        #print(df_c)
            #df_c = df_c.iloc[:-1, :]   
        dfs_c.append(df_c) 
       # print(dfs_c)       
    df = pd.concat(dfs_c, axis=0, ignore_index=True)
      
    df = df.replace(-999.999, np.NaN)        
 #   idx2 = pd.date_range(start = pd.Timestamp(date1), \
  #                                    end = pd.Timestamp(date2), freq='H')  
    df = df.set_index(idx1)

    df = df.drop(columns=[0, 1, 2, 3, 4])
    df = df.rename(columns={5:'gic', 6:'T1', 7:'T2'})
    #gic = df.iloc[:,5]
    #T1  = df.iloc[:,6]
    #T2  = df.iloc[:,7]
    output = {}
    output.update({stat:df})
    return(output)

###############################################################################
###############################################################################
def df_dH(date1, date2, dir_path, H_stat):
    idx1 = pd.date_range(start = pd.Timestamp(date1), end =\
                         pd.Timestamp(date2+' 23:00:00'), freq='H')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                                          end = pd.Timestamp(date2), freq='D')
        
    idx_list = (idx_daylist.strftime('%Y%m%d')) 
    str1 = str(H_stat)+"_"
    ext = ".delta_H.early"
    station = ''
    if H_stat == 'coe':
       station =  'coeneo'
    elif H_stat == 'teo':
        station = 'teoloyucan'
    elif H_stat == 'itu':
        station = 'iturbide'

    remote_path= '/data/output/indexes/'+station+'/'
    list_fnames = list_names(idx_list, str1, ext)
    wget = get_files(date1, date2, remote_path, dir_path, list_fnames)
    dfs_c = []
        
    for file_name in list_fnames: 
  #  for file_name in file_names:    
        df_c = pd.read_csv(dir_path+file_name, header=None, sep='\s+', \
                           skip_blank_lines=True).T
        df_c = df_c.iloc[:-1, :]   
        dfs_c.append(df_c) 
                
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    
    df = df.replace(999999.0, np.NaN)        
 #   idx2 = pd.date_range(start = pd.Timestamp(date1), \
  #                                    end = pd.Timestamp(date2), freq='H')
        
    df = df.set_index(idx1)
    
    df = df.loc[date1:date2]
    H  = df.iloc[:,0]

    return(H)

###############################################################################
###############################################################################
def df_Kloc(date1, date2, dir_path):
  #  dir_path = '/home/isaac/MEGAsync/datos/Kmex/coe'

    idx1 = pd.date_range(start = pd.Timestamp(date1), end = \
                             pd.Timestamp(date2+' 21:00:00'), freq='3H')

    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                  end = pd.Timestamp(date2), freq='D')
                
    idx_list = (idx_daylist.strftime('%Y%m%d')) 
        
    str1 = "coe_"
    ext = ".k_index.early"
    remote_path= '/data/output/indexes/coeneo/'
    
    list_fnames = list_names(idx_list, str1, ext)
    wget = get_files(date1, date2, remote_path, dir_path, list_fnames)

    dfs_c = []
            
    for file_name in list_fnames:    
        df_c = pd.read_csv(dir_path+file_name, header=None, sep='\s+', \
                           skip_blank_lines=True).T
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

def fix_offset(f):
    
    f_med = f.median()
    
    f_baseline = np.repeat(f_med, len(f))
    
    #baseline_offset = np.linspace(0, len(gicTW_lav))
    f_fixed = f-f_baseline    
    
    return(f_fixed)