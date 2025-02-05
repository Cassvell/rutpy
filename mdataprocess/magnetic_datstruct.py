#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:28:00 2024

@author: isaac
"""
import pandas as pd
import os
import numpy as np
from despike import despike

def get_dataframe(filenames, path, idx, daily_idx, net):       
    dfs_c = []
    filenames = sorted(filenames)
    
    H = []
    if net == 'regmex':
        for i in range(len(filenames)):
        
            # Construct the full path
            full_path = os.path.join(path, filenames[i])
            
            # Check if the file exists
            if os.path.isfile(full_path):
                try:
                    # Read the file into a dataframe
                    df_c = pd.read_csv(full_path, header=None, delim_whitespace=True)
                    dfs_c.append(df_c)
                except Exception as e:
                    # Handle any read errors
                    print(f"Error reading {full_path}: {e}")
            else:
                print(f"File not found: {full_path}")
                dailyfile = pd.date_range(start = pd.Timestamp(str(daily_idx[i])), \
                                periods=1440, freq='T')
                df_c = np.empty((1440, 10), dtype=object)
                
                df_c[:, 0] = dailyfile.day.astype(str).str.zfill(2)       # Day (DD)
                df_c[:, 1] = dailyfile.month.astype(str).str.zfill(2)     # Month (MM)
                df_c[:, 2] = dailyfile.year.astype(str)                   # Year (YYYY)
                df_c[:, 3] = dailyfile.hour.astype(str).str.zfill(2)      # Hour (HH)
                df_c[:, 4] = dailyfile.minute.astype(str).str.zfill(2)    # Minute (MM)
                #Fill the last 5 columns with np.nan
                df_c[:, 5:] = np.nan               
                df_c = pd.DataFrame(df_c)                
                
        # Concatenate all dataframes in the list
        if dfs_c:
            final_df = pd.concat(dfs_c, ignore_index=True)
            print("Dataframe concatenated successfully.")
        else:
            print("No valid dataframes to concatenate.")
        #   else:
    ###############################################################################
    #COMBINE THE DAILY DATAFRAMES        
        # dfs_c.append(df_c) 
            
        df = pd.concat(dfs_c, axis=0, ignore_index=True)    
        df = df.replace(999999.00, np.NaN) 
        df = df.replace(9999.00, np.NaN)      
    
        #df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d')
    # Combine Date and Time columns as strings
        df['DateTime'] =  df.iloc[:, 0]+ ' ' + df.iloc[:, 1]
        df['DateTime'] = pd.to_datetime(df['DateTime'])
                        
    #df = df.dropna(subset=[df.columns[0], df.columns[1]], axis=1)    

        df = df.set_index(df['DateTime'])   
        df = df.reindex(idx)   
        df = df.drop(columns=[0, 1, 'DateTime'])
        
        H = df.iloc[:,2]    
        D = df.iloc[:,1]
        Z = df.iloc[:,3]
        F = df.iloc[:,4]

        Ddeg = D/60
        deg2rad = np.pi / 180
        D = deg2rad*Ddeg
        X = H*np.cos(D)
        Y = H*np.sin(D)
        I = np.tan(Z/H)

        H = despike(H, threshd = 7.5)
        X = despike(X, threshd = 7.5)
        Y = despike(Y, threshd = 7.5)
        Z = despike(Z, threshd = 7.5)

        for i in range(len(H)):
            if H[i] > 60000:
                H[i] = np.nan
            if X[i] > 60000:
                X[i] = np.nan    

        data = {'H' : pd.Series(H), 'X' : pd.Series(X), 'Y' : pd.Series(Y), 'Z' : pd.Series(Z)}
        df_2 = pd.DataFrame(data)
        df_2 = df_2.set_index(idx)
        df_2['D'] = D
        df_2['I'] = I


    else: #si se trata de intermagnet
        for i in range(len(filenames)):
        
            # Construct the full path
            full_path = os.path.join(path, filenames[i])
            # Check if the file exists
            if os.path.isfile(full_path):
                try:
                    date_line_number = None

                    with open(full_path, 'r') as file:
                        for line_number, line in enumerate(file, start=0):
                            if 'DATE' in line:
                                date_line_number = line_number
                                break
        
                    # Read the file into a dataframe
                    df_c = pd.read_csv(full_path, header=date_line_number, delim_whitespace=True)
                    dfs_c.append(df_c)
                except Exception as e:
                    # Handle any read errors
                    print(f"Error reading {full_path}: {e}")
            else:
                print(f"File not found: {full_path}")
                dailyfile = pd.date_range(start = pd.Timestamp(str(daily_idx[i])), \
                                periods=1440, freq='T')
                df_c = np.empty((1440, 10), dtype=object)
                
        #        df_c[:, 0] = dailyfile.day.astype(str).str.zfill(2)       # Day (DD)
        #        df_c[:, 1] = dailyfile.month.astype(str).str.zfill(2)     # Month (MM)
        #        df_c[:, 2] = dailyfile.year.astype(str)                   # Year (YYYY)
        #        df_c[:, 3] = dailyfile.hour.astype(str).str.zfill(2)      # Hour (HH)
        #        df_c[:, 4] = dailyfile.minute.astype(str).str.zfill(2)    # Minute (MM)
                #Fill the last 5 columns with np.nan
        #        df_c[:, 5:] = np.nan               
                df_c = pd.DataFrame(df_c)                
                
        # Concatenate all dataframes in the list
        if dfs_c:
            final_df = pd.concat(dfs_c, ignore_index=True)
            print("Dataframe concatenated successfully.")
        else:
            print("No valid dataframes to concatenate.")
    ###############################################################################
    #COMBINE THE DAILY DATAFRAMES        
            
        df = pd.concat(dfs_c, axis=0, ignore_index=True)    
        df = df.replace(999999.00, np.NaN) 
        df = df.replace(9999.00, np.NaN)      
            
        df['DateTime'] =  df.iloc[:, 0]+ ' ' + df.iloc[:, 1]
        df['DateTime'] = pd.to_datetime(df['DateTime'])
                        

        df = df.set_index(df['DateTime'])   
        
        df = df.reindex(idx)   
        df = df.drop(columns=['DATE', 'TIME', 'DOY', 'DateTime', '|'])

        H = np.sqrt(df.iloc[:,0]**2   + df.iloc[:,1]**2)
        X = df.iloc[:,0]
        Y = df.iloc[:,1]
        Z = df.iloc[:,2]
        D = np.tan(Y / X)
        I = np.tan(Z / H)
        
        
        H = despike(H, threshd = 7.5)
        X = despike(X, threshd = 7.5)
        Y = despike(Y, threshd = 7.5)
        Z = despike(Z, threshd = 7.5)

        for i in range(len(H)):
            if H[i] > 60000:
                H[i] = np.nan
            if X[i] > 60000:
                X[i] = np.nan    

        data = {'H' : pd.Series(H), 'X' : pd.Series(X), 'Y' : pd.Series(Y), 'Z' : pd.Series(Z)}
        df_2 = pd.DataFrame(data)
        df_2 = df_2.set_index(idx)
        df_2['D'] = D
        df_2['I'] = I
        
    return df_2