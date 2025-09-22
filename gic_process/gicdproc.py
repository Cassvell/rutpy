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
import sys
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


from get_files import get_files, list_names, get_file

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
def process_station_data(i_date, f_date, path2, stat, idx1, tot_data):
    # Create the daily index range
    daily_index = pd.date_range(
        start=pd.Timestamp(i_date),
        end=pd.Timestamp(f_date + " 23:59:00"),
        freq="D"
    )

    # Initialize the file_exists flag
    # Loop over daily_index to check if any file exists
    SG2 = []
    for i in range(len(daily_index)):
        year = daily_index[i].year  
        date_str = daily_index[i].strftime("%Y-%m-%d")
        
        SG2_tmp = f"{path2}{year}/{stat}/daily/GIC_{date_str}_{stat}.dat"
        SG2.append(SG2_tmp)
       # print(SG2)
        #if os.path.isfile(SG2_tmp):
         #   print('exist')
 

    if os.path.isfile(SG2[0]):
        df = df_gic(i_date, f_date, path2, stat)

        # Ensure df is a dictionary and contains the key stat
        if isinstance(df, dict) and stat in df:
            gic_data = df[stat].loc[:, "gic"]
            gic_data = fix_offset(gic_data)
            T1_data = df[stat].loc[:, "T1"]
            T2_data = df[stat].loc[:, "T2"]
        else:
            raise ValueError(f"Data for station {stat} not found in df_gic output.")

    else:
        # Ensure idx1 is defined before using it
        if 'idx1' not in locals():
            raise ValueError("idx1 is not defined before using it as an index!")

        # If no file exists, return NaN-filled DataFrame
        df = pd.DataFrame(
            np.full((tot_data, 3), np.nan),
            columns=["gic", "T1", "T2"],
            index=idx1  # Ensure idx1 is valid
        )

        gic_data = df["gic"]
        T1_data = df["T1"]
        T2_data = df["T2"]
        
    return gic_data, T1_data, T2_data

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
                         pd.Timestamp(nextday+' 12:00:00'), freq='min')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(str(date1)), \
                                          end = pd.Timestamp(str(date2)), freq='D')
        
    idx_list = (idx_daylist.strftime('%Y-%m-%d')) 
    str1 = "GIC_"
    ext = "_"+stat+".dat"

   # remote_path= '/data/output/indexes/'+station+'/'
    list_fnames = list_names(idx_list, str1, ext)

    dfs_c = []
    missing_vals = ["NaN", "NO DATA"] 

    for i in range(len(list_fnames)):      
        year = idx_daylist[i].year  # Extract year directly from the Timestamp
        SG2 = f"{dir_path}{year}/{stat}/daily/"
        
        
        try:
            # Read first 5 rows
            df_c = pd.read_csv(SG2+list_fnames[i], header=0, sep='\t',parse_dates = [0], na_values = missing_vals)
            
            # Count empty values
            empty_counts = df_c.isna().sum()  # Count NaN values
            empty_strings = (df_c == '').sum()  # Count empty strings
            total_empty = empty_counts + empty_strings
            
            # Check if any empty values exist (sum across all columns)
            if total_empty.sum() > 0:
                print(f"Found {total_empty.sum()} empty values to replace")
                
                # Replace both NaN and empty strings
                df_c.replace({'': -999.999}, inplace=True)
                df_c.fillna(-999.999, inplace=True)
                
                # Save back to same file (consider using a different filename for safety)
                #output_filename = filename
                #df.to_csv(path + output_filename, sep='\t', index=False, na_rep='-999.999')
                #print(f"File saved with empty values replaced: {path + output_filename}")
            else:
                print("No empty values found in first 5 rows")
                
        except FileNotFoundError:
            print(f"Error: File not found - {SG2+list_fnames[i]}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
        
        
        #df_c = pd.read_csv(SG2+list_fnames[i], header=None, skiprows = 1, sep='\s+', parse_dates = [0], na_values = missing_vals)
        
            #df_c = df_c.iloc[:-1, :]   
        dfs_c.append(df_c)     
          
    df = pd.concat(dfs_c, axis=0, ignore_index=True)
      
    df = df.replace(-999.999, np.nan)        
 #   idx2 = pd.date_range(start = pd.Timestamp(date1), \
  #                                    end = pd.Timestamp(date2), freq='H')  
    df = df.set_index(idx1)

    #df = df.drop(columns=[0, 1, 2, 3, 4])
    df = df.rename(columns={5:'gic', 6:'T1', 7:'T2'})
    #gic = df.iloc[:,5]
    #T1  = df.iloc[:,6]
    #T2  = df.iloc[:,7]
    output = {}
    output.update({stat:df})
    #return(df)
    return(output)

def df_gic_daily(date, dir_path, stat):
    col_names = ['Datetime','gic', 'T1','T2', 'gic_proc', 'T1_proc',	'T2_proc']
    
    fyear = int(date[0:4])
    fmonth = int(date[4:6])
    fday = int(date[6:8])


    finaldate= datetime(fyear, fmonth,fday)
    nextday = finaldate+timedelta(days=1)
    nextday = str(nextday)[0:10]
    
    
    idx1 = pd.date_range(start = pd.Timestamp(str(date)+' 12:01:00' ), end =\
                         pd.Timestamp(nextday+' 12:00:00'), freq='T')
        
    idx_list = (date.strftime('%Y-%m-%d')) 
    str1 = "GIC_"
    ext = "_"+stat+".dat"

   # remote_path= '/data/output/indexes/'+station+'/'

    missing_vals = ["NaN", "NO DATA"] 
    
    year = idx_list.year  # Extract year directly from the Timestamp
    SG2 = f"{dir_path}{year}/{stat}/daily/"
      
    df = pd.read_csv(SG2+list_fnames[i], header=None, skiprows = 1, sep='\\s+', parse_dates = [0], na_values = missing_vals)
         
    df = pd.concat(dfs_c, axis=0, ignore_index=True)
      
    df = df.replace(-999.999, np.nan)        
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
                         pd.Timestamp(date2+' 23:00:00'), freq='h')
    
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
    dfs_c = []
    for filename in list_fnames:
        if os.path.exists(dir_path+filename): #si el archivo diario ya está en la carpeta local, leelo
            df_c = pd.read_csv(dir_path+filename, header=None, sep=r'\s+', skip_blank_lines=True).T
            df_c = df_c.iloc[:-1, :]   

        else:
            wget = get_file(remote_path, dir_path, filename) #si no está, búscalo en la carpeta remota, copialo a la local y leelo
            if wget == True:
                df_c = pd.read_csv(dir_path+filename, header=None, sep=r'\s+', skip_blank_lines=True).T
                df_c = df_c.iloc[:-1, :]   
            else: #si tampoco está en la local, genera un dataframe de archivos nulos
                
                        df_c = pd.DataFrame(
                                            np.full((24, 2), np.nan)
                                        )             
        
        dfs_c.append(df_c) #combina los dataframes, incluyendo los vacíos
        
 
        
    df = pd.concat(dfs_c, axis=0, ignore_index=True)  

    df = df.replace(999999.0, np.nan)        
    idx2 = pd.date_range(start = pd.Timestamp(date1), \
                                    end = pd.Timestamp(date2), freq='h')
    df = df.set_index(idx1)
    
    df = df.loc[date1:date2]
    H  = df.iloc[:,0]
    
    return(H)

###############################################################################
def df_sym(date1, date2, dir_path):
    idx1 = pd.date_range(start = pd.Timestamp(date1), end =\
                         pd.Timestamp(date2+' 23:59:00'), freq='min')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                                          end = pd.Timestamp(date2), freq='D')  

    str1 = 'sym_'
    ext = 'm_D.dat'

    list_fnames = []
    for i in idx_daylist:
        date_str = str(i)[0:10]  # Get yyyy-mm-dd format
        
        # First try with m_D.dat
        tmp = str1 + date_str + ext
        if os.path.isfile(f'{dir_path}{tmp}'):
            list_fnames.append(tmp)
            continue  # Move to next date if found
        
        # If not found, try m_P.dat
        ext = 'm_P.dat'
        tmp = str1 + date_str + ext
        if os.path.isfile(f'{dir_path}{tmp}'):
            list_fnames.append(tmp)
            continue  # Move to next date if found
        
        # If not found, try m_Q.dat
        ext = 'm_Q.dat'
        tmp = str1 + date_str + ext
        if os.path.isfile(f'{dir_path}{tmp}'):
            list_fnames.append(tmp)
        else:
            print(f'{tmp} file does not exist')

    dfs_c = []
    
    for file_name in list_fnames:    
        df_c = pd.read_csv(f'{dir_path}{file_name}', header=None, sep='\\s+', skip_blank_lines=True)
        
        dfs_c.append(df_c) 
                
    df = pd.concat(dfs_c, axis=0, ignore_index=True)
    
    
    index = {'DateTime' : idx1, 'ASYD' : df.iloc[:,0], 'ASYH' : df.iloc[:,1], 'SYMH' : df.iloc[:,2], 'SYMD' : df.iloc[:,3]}
    
    return(index)

###############################################################################
def df_dst(date1, date2, dir_path):
    idx1 = pd.date_range(start = pd.Timestamp(date1), end =\
                         pd.Timestamp(date2+' 23:00:00'), freq='H')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                                          end = pd.Timestamp(date2), freq='D')  
    
    str1 = 'dst_'
    ext = '.dat'
    list_fnames = []
    for i in idx_daylist:
        tmp = str1+str(i)[0:10]+ext
        list_fnames.append(tmp)
    
    dfs_c = []
        
    for file_name in list_fnames:    
        df_c = pd.read_csv(dir_path+file_name, header=0, delim_whitespace=True, skip_blank_lines=True)
        
        dfs_c.append(df_c) 
                
    df = pd.concat(dfs_c, axis=0, ignore_index=True)
        
    df = df.set_index(idx1)
    return(df['Dst'])


def df_dHmex(date1, date2, dir_path, H_stat):
    idx1 = pd.date_range(start = pd.Timestamp(date1), end =\
                         pd.Timestamp(date2+' 23:00:00'), freq='h')
    
    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                                          end = pd.Timestamp(date2), freq='D')
        
    idx_list = (idx_daylist.strftime('%Y%m%d')) 
    str1 = str(H_stat)+"_"
    ext = ".delta_H.early"
    station = ''
    if H_stat == 'mex':
       station =  'mexico'
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
        df_c = pd.read_csv(dir_path+file_name, header=None, sep=r'\s+', skip_blank_lines=True).T
        df_c = df_c.iloc[:-1, :]   
        dfs_c.append(df_c) 
                
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    
    df = df.replace(999999.0, np.nan)        
 #   idx2 = pd.date_range(start = pd.Timestamp(date1), \
  #                                    end = pd.Timestamp(date2), freq='H')
        
    df = df.set_index(idx1)
    
    df = df.loc[date1:date2]
    H  = df.iloc[:,0]

    return(H)
###############################################################################
###############################################################################
def df_Kloc(date1, date2, dir_path, stat):
  #  dir_path = '/home/isaac/MEGAsync/datos/Kmex/coe'

    idx1 = pd.date_range(start = pd.Timestamp(date1), end = \
                             pd.Timestamp(date2+' 21:00:00'), freq='3h')

    idx_daylist = pd.date_range(start = pd.Timestamp(date1), \
                  end = pd.Timestamp(date2), freq='D')
                
    idx_list = (idx_daylist.strftime('%Y%m%d')) 
        
    str1 = f"{stat}_"
    ext = ".k_index.early"
    station = ''
    if stat == 'mex':
       station =  'mexico'
    elif stat == 'teo':
        station = 'teoloyucan'
    elif stat == 'itu':
        station = 'iturbide'
    remote_path= f'/data/output/indexes/{station}/'
    
    list_fnames = list_names(idx_list, str1, ext)
    dfs_c = []
    for filename in list_fnames:
        if os.path.exists(dir_path+filename): #si el archivo diario ya está en la carpeta local, leelo
            df_c = pd.read_csv(dir_path+filename, header=None, sep=r'\s+', skip_blank_lines=True).T
            df_c = df_c.iloc[:-1, :]   

        else:
            wget = get_file(remote_path, dir_path, filename) #si no está, búscalo en la carpeta remota, copialo a la local y leelo
            if wget == True:
                df_c = pd.read_csv(dir_path+filename, header=None, sep=r'\s+', skip_blank_lines=True).T
                df_c = df_c.iloc[:-1, :]   
            else: #si tampoco está en la local, genera un dataframe de archivos nulos
                
                        df_c = pd.DataFrame(
                                            np.full((8, 6), 999)
                                        )             
        
        dfs_c.append(df_c) #combina los dataframes, incluyendo los vacíos
#        print(dfs_c)

          
            
   # idx2 = pd.date_range(start = pd.Timestamp(date1), \
    #                                      end = pd.Timestamp(date2), freq='3H')
    df = pd.concat(dfs_c, axis=0, ignore_index=True)    
    df = df.set_index(idx1)
        
    df = df.loc[date1:date2]
    
    k  = df.iloc[:,0]
    #print(k)

    
    k = k/10
    return(k)

def fix_offset(f):
    
    f_med = f.median()
    
    f_baseline = np.repeat(f_med, len(f))
    
    #baseline_offset = np.linspace(0, len(gicTW_lav))
    f_fixed = f-f_baseline    
    
    return(f_fixed)