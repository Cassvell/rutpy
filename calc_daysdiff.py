#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:04:26 2023

@author: isaac
"""

import sys
from datetime import datetime

def calculate_days_difference(date_str1, date_str2):
    # Convert date strings to datetime objects
    date1 = datetime.strptime(date_str1, '%Y%m%d')
    date2 = datetime.strptime(date_str2, '%Y%m%d')

    # Calculate the difference in days
    days_difference = abs((date2 - date1).days)

    return days_difference

if __name__ == "__main__":
    # Check if two date strings are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py yyyymmdd1 yyyymmdd2")
        sys.exit(1)

    date_str1 = sys.argv[1]
    date_str2 = sys.argv[2]

    try:
        # Calculate and print the number of days between the two dates
        days_difference = calculate_days_difference(date_str1, date_str2)
        print(f"Number of days between {date_str1} and {date_str2}: {days_difference} days")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
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



df_qro = pproc('QRO', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/QRO/')
df_lav = pproc('LAV', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/LAV/')
df_rmy = pproc('RMY', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/RMY/')
df_mzt = pproc('MZT', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/MZT/')



df_lav = df_gic(i_date, f_date,'/home/isaac/MEGAsync/datos/gics_obs/2023/LAV/daily/', 'LAV')
df_rmy = df_gic(i_date, f_date,'/home/isaac/MEGAsync/datos/gics_obs/2023/RMY/daily/', 'RMY')
df_mzt = df_gic(i_date, f_date,'/home/isaac/MEGAsync/datos/gics_obs/2023/MZT/daily/', 'MZT')
df_qro = df_gic(i_date, f_date,'/home/isaac/MEGAsync/datos/gics_obs/2023/QRO/daily/', 'QRO')


  
gicTW_lav = df_lav['LAV'].gic
gicTW_lav = fix_offset(gicTW_lav)

gicTW_qro = df_qro['QRO'].gic
gicTW_rmy = df_rmy['RMY'].gic
gicTW_mzt = df_mzt['MZT'].gic


T1TW_lav = df_lav['LAV'].T1
T1TW_qro = df_qro['QRO'].T1
T1TW_rmy = df_rmy['RMY'].T1
T1TW_mzt = df_mzt['MZT'].T1

T2TW_lav = df_lav['LAV'].T2
T2TW_qro = df_qro['QRO'].T2
T2TW_rmy = df_rmy['RMY'].T2
T2TW_mzt = df_mzt['MZT'].T2
"""