#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:52:48 2024
@author: isaac
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from magnetic_datstruct import get_dataframe
from aux_time_DF import index_gen, convert_date

from datetime import datetime, date, timedelta
from magdata_processing import get_qd_dd, max_IQR, base_line

###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
###############################################################################
st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
fdate = sys.argv[3]

idate = datetime.strptime(idate + ' 00:00:00', '%Y%m%d %H:%M:%S')
fdate = datetime.strptime(fdate + ' 23:59:00', '%Y%m%d %H:%M:%S')

iw = fdate - timedelta(days=26)-timedelta(hours=23)-timedelta(minutes=59)
###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')                        
fw_dates = []
#print(idx_daily)


path = '/home/isaac/datos/regmex/'+st+'/'
path_qdl = '/home/isaac/geomstorm/rutpy/mdataprocess/qdl_training' 

df = pd.read_excel(path_qdl+'qdl.ods', engine='odf', sheet_name=0)
column = df.columns[1:-1]
filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)
    
data = get_dataframe(filenames, path, idx, dates)

#baseline, trend, DD = base_line(data, idx, idx_daily)

#H_det[np.isnan(H_det)] = 999.9   
status_array = np.zeros((len(idx_daily), 27))
data_array = np.full((len(idx_daily), 27, 1440), 999.9)
qdl_list_per_w = [[''] * 5 for _ in range(len(idx_daily))]


C = np.zeros((len(idx_daily), 1440))
stat = np.zeros((len(idx_daily), 27))
zero_entries = []  # List to store indices with zero values
for i, col in enumerate(column):
    fw = idx_daily[i] + timedelta(days=26)
    fw_dates.append(fw)

    tmp_index = pd.date_range(start = pd.Timestamp(str(idx_daily[i])), \
                            end = pd.Timestamp(str(fw)), freq='D')
  

    tmp_inicio = str(idx_daily[i])[0:10]
    
    tmp_final= str(fw)[0:10]

    data_per_window = data[tmp_inicio:tmp_final]

    iqr_picks = max_IQR(data_per_window, 60, 24)
    iqr_picks = np.array(iqr_picks)

    valid_days = iqr_picks[~np.isnan(iqr_picks)]

    #list_of_dates = get_qd_dd(iqr_picks, tmp_index, 'qdl', len(tmp_index))
    list_of_dates = str(tmp_index)[0:10]
    
    Ddates = []
    #if qd = qd[j] = df[i].iloc[j]
    np.full(((len(df.columns)-1), 10, 1440), 999.9)
    days_of_month = np.full((len(tmp_index)), 1)#np.zeros(len(tmp_index))
            
    tmp = {'date' : tmp_index, 'status' : days_of_month}
    
    monthly_status = pd.DataFrame(tmp)

    A = np.array(monthly_status['status'])

    M = [[0] * 1440 for _ in range(27)]
    
    if len(valid_days) > 10:
        #qdl_list_per_w[i] = get_qd_dd(iqr_picks, tmp_index, 'qdl', 5)
        qdl = [[0] * 1440 for _ in range(10)]
        qd = [0] * 10
        baseline = []
        
        for j in range(10):
            #print(qd[j])
            qd[j] = df[col].iloc[j]  #str(qdl_list_per_w[i][j])[:10]  # Get the date as string yyyy-mm-dd
            
            if qd[j] is not pd.NaT:
                monthly_status.loc[monthly_status['date'].isin([qd[j]]), 'status'] = 0
               # qdl[j] = qd_arr.reset_index(drop=True)
                qd[j] = str(qd[j])[0:10]
                #print(data[qd[j]])
                qdl[j] = data.get(qd[j], None)
                
                if qdl[j] is not None:
                    #print(qdl[j])
                    #qdl[j] = qdl[j].reset_index(drop=True)
                    
                    if len(qdl[j]) >= 480:  
                        qd_2h = qdl[j][300:480]
                        
                        qdl[j] = qdl[j] - np.nanmedian(qd_2h)
                    
                #qd_2h = qdl[j][60:300]  # Get the first 2 hours (assuming 1 minute resolution)

        baseline =  np.nanmedian(qd_2h)  # Adjust the data by subtracting the median
        
        H_det = data_per_window-baseline
        
        H_det[np.isnan(H_det)] = 999.9   
        
        tmp_data = np.array(H_det)   

        
        for j in range(27):
            data_array[i, j, :] = tmp_data[j*1440:(j+1)*1440]
            status_array[i, j] = A[j] if j < len(A) else 0  # Ensure no out of range errors  

    else:

        A = np.full(len(tmp_index), 0)
        for j in range(27):
          #  data_array[i, j, :] = [np.nan] * 1440
            data_array[i, j, :] = [999.9] * 1440
            status_array[i, j] = A[j]

    #monthly_status.loc[monthly_status['date'].isin(Ddates), 'status'] = 1

     
    #plt.plot(data_array[1, j, :])
    print(status_array)

    if idx_daily[i] >= iw:
        
        break  
#plt.show()

np.save('/home/isaac/test_isaac/X_huan_2023_24.npy', data_array)
np.save('/home/isaac/test_isaac/Y_huan_2023_24.npy', status_array)


#print(M)
#plt.show()
