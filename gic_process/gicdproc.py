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
import warnings
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
    
    #import matplotlib.pyplot as plt
    #plt.plot(gic_data)
    #plt.show()
    return gic_data, T1_data, T2_data

###############################################################################
###############################################################################
def skip_initial_empty_rows(file_path, delimiter=None):
    """Skip rows until we find one with substantial content"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Use the same whitespace splitting logic as pd.read_csv(sep='\\s+')
        if delimiter:
            # If specific delimiter provided
            non_empty_fields = [field for field in line.strip().split(delimiter) if field.strip()]
        else:
            # Use whitespace splitting (multiple spaces/tabs)
            non_empty_fields = line.strip().split()
        
        #print(f"Row {i}: {len(non_empty_fields)} columns - {non_empty_fields}")
        
        if len(non_empty_fields) >= 8:  # Adjust based on your expected column count
            row = i
            break
            
        #    return i
        else:
            row = 0
            
            
    return row

def df_gic(date1, date2, dir_path, stat):
    col_names = ['Datetime','gic', 'T1','T2', 'gic_proc', 'T1_proc',	'T2_proc']
    
    fyear = int(date2[0:4])
    fmonth = int(date2[4:6])
    fday = int(date2[6:8])


    finaldate= datetime(fyear, fmonth,fday)
    nextday = finaldate+timedelta(days=1)
    nextday = str(nextday)[0:10]
    
    
    idx1 = pd.date_range(start = pd.Timestamp(str(date1)+' 12:00:00' ), end =\
                         pd.Timestamp(nextday+' 11:59:00'), freq='min')
    
    
    idx_daylist = pd.date_range(start = pd.Timestamp(str(date1)), \
                                          end = pd.Timestamp(str(date2)), freq='D')
        

    idx_list = (idx_daylist.strftime('%Y-%m-%d')) 
    str1 = "GIC_"
    ext = "_"+stat+".dat"

   # remote_path= '/data/output/indexes/'+station+'/'
    list_fnames = list_names(idx_list, str1, ext)

    dfs_c = []
    missing_vals = ["NaN", "NO DATA"] 
    column_names = ['Datetime', 'gic', 'T1', 'T2', 'gic_proc', 'T1_proc', 'T2_proc']
    for i in range(len(list_fnames)):      
        year = idx_daylist[i].year
        SG2 = f"{dir_path}{year}/{stat}/daily/"
        
        try:
            # Construct full file path
            file_path = os.path.join(SG2, list_fnames[i])
            
            # Check if file exists
            if os.path.isfile(file_path):
                
                
                file_path = file_path
                skip_rows = skip_initial_empty_rows(file_path)             

                tmp_idx = idx1[i*1440: (i+1)*1440]

                # Then use the appropriate format
                column_names_modified = ['date', 'time'] + column_names[1:]
                df_c = pd.read_csv(file_path, header=None,skiprows=skip_rows, sep='\\s+', 
                                na_values=missing_vals, keep_default_na=True, names=column_names_modified)
                
                
                #datetime_parts = df_c.iloc[:, 0].str.extract(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})')
                df_c['Datetime'] = pd.to_datetime(df_c['date'] + ' ' + df_c['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                
                df_c = df_c.drop(['date', 'time'], axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    
                    if df_c.isna().any().any():
                        numeric_cols = df_c.select_dtypes(include=[np.number]).columns
                        object_cols = df_c.select_dtypes(include=['object']).columns
                        
                        if len(numeric_cols) > 0:
                            df_c[numeric_cols] = df_c[numeric_cols].fillna(-999.999)

                      
                df_c = df_c.set_index(df_c['Datetime'])
                df_c = df_c[~df_c.index.duplicated(keep='first')]
                df_c = df_c.reindex(tmp_idx)
                
                #df_c = df_c.set_index(tmp_idx)
                
                df_c = df_c.drop(columns=['Datetime', 'gic', 'T1', 'T2'])
                df_c = df_c.rename(columns={'gic_proc':'gic', 'T1_proc':'T1', 'T2_proc':'T2'})          
       
            else: 
                # If no file exists, create consistent DataFrame structure
                # Define idx1 appropriately - you'll need to define this variable
                #idx1 = ...  # Define your timestamp index here
                
                df_c = pd.DataFrame({
                    'gic_proc': np.full(1440, -999.999), 'T1_proc': np.full(1440, -999.999),
                    'T2_proc': np.full(1440,-999.999)})
                df_c = df_c.set_index(tmp_idx)

        except FileNotFoundError:
            print(f"Error: File not found - {SG2+list_fnames[i]}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
  
        dfs_c.append(df_c)     

    df = pd.concat(dfs_c, axis=0)

    df = df.replace(-999.999, np.nan)        

    output = {}
    output.update({stat:df})
    return(output)

def df_gic_pp(date1, date2, dir_path, stat):
    col_names = ['Datetime','gic', 'T1','T2']
    
    
    
    idx1 = pd.date_range(start = pd.Timestamp(str(date1)+' 00:00:00' ), end =\
                         pd.Timestamp(str(date2)+' 23:59:00'), freq='min')
    
    
    idx_daylist = pd.date_range(start = pd.Timestamp(str(date1)), \
                                          end = pd.Timestamp(str(date2)), freq='D')
    idx_list = (idx_daylist.strftime('%Y-%m-%d')) 
    str1 = f"{stat}_"
    ext = ".pp.csv"

   # remote_path= '/data/output/indexes/'+station+'/'
    list_fnames = list_names(idx_list, str1, ext)

    dfs_c = []
    column_names = ['Datetime', 'gic', 'T1', 'T2']
    for i in range(len(list_fnames)):      
        year = idx_daylist[i].year
        
        try:
            # Construct full file path
            file_path = os.path.join(dir_path, str(year), stat, "daily", list_fnames[i])
            
            # Check if file exists
            if os.path.isfile(file_path):
                
                tmp_idx = idx1[i*1440: (i+1)*1440]

                # Read data
                #column_names_modified = ['date', 'time'] + column_names[1:]
                df_c = pd.read_csv(file_path, header=0, sep='\t')
                #print(df_c)
                #sys.exit('end')
                # Create datetime index
                df_c['Datetime'] = pd.to_datetime(df_c['Datetime'], 
                                                format='%Y-%m-%d %H:%M:%S', errors='coerce')
                #df_c = df_c.drop(['date', 'time'], axis=1)

                #dt = datetime.datetime.fromtimestamp(df_c['Datetime'])
                
                
                
                # Set index and handle duplicates
                df_c = df_c.set_index(df_c['Datetime'])
                df_c = df_c[~df_c.index.duplicated(keep='first')]
                df_c = df_c.replace(999.9, np.nan)
                
            else: 
                # Create consistent empty DataFrame
                tmp_idx = idx1[i*1440: (i+1)*1440]
                # Use the actual column structure from your data
                empty_data = {col: np.full(1440, np.nan) for col in column_names[2:]}
                df_c = pd.DataFrame(empty_data)
                df_c = df_c.set_index(tmp_idx)

        except Exception as e:
            print(f"Error processing file {list_fnames[i]}: {str(e)}")
            # Create empty DataFrame on error too
            tmp_idx = idx1[i*1440: (i+1)*1440]
            empty_data = {col: np.full(1440, np.nan) for col in column_names[2:]}
            df_c = pd.DataFrame(empty_data)
            df_c = df_c.set_index(tmp_idx)
    
        dfs_c.append(df_c)     

    # Final processing
    df = pd.concat(dfs_c, axis=0)   
    
    
    return(df)


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
    elif stat == 'coe':
        station == 'coeneo'
    elif stat == 'teo':
        station = 'teoloyucan'
    elif stat == 'itu':
        station = 'iturbide'
    remote_path= f'/data/output/indexes/{station}/'
    
    list_fnames = list_names(idx_list, str1, ext)
    dfs_c = []
    for filename in list_fnames:
        #print(filename)
        if os.path.exists(dir_path+filename): #si el archivo diario ya está en la carpeta local, leelo
            #print(dir_path+filename)
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