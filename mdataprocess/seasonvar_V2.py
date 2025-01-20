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
from scipy.interpolate import splrep, splev
from magnetic_datstruct import get_dataframe
from aux_time_DF import index_gen, convert_date

from datetime import datetime, date, timedelta
from magdata_processing import get_qd_dd, max_IQR
from scipy.interpolate import interp1d
from Ffitting import fit_data
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

ventana_final = fdate - timedelta(days=10)-timedelta(hours=23)-timedelta(minutes=59)
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

###############################################################################
###############################################################################
#FOURIER FIT
###############################################################################
###############################################################################

from magdata_processing import get_qd_dd, max_IQR, base_line

path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 

def fill_gap(data):

        def nan_helper(y):    
            return np.isnan(y), lambda z: z.nonzero()[0]   
        
        nans, x = nan_helper(data)    
        
        valid_data_indices = x(~nans)
        valid_data_values = data[valid_data_indices]
        
        cubic_interp = interp1d(valid_data_indices, valid_data_values, kind='cubic', fill_value="extrapolate")
        
        data[nans] = cubic_interp(x(nans))
        
        return data

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
ndata = len(data)
lenght_window = 10   
nweeks = 31

qdl_list_per_w = [[''] * 4 for _ in range(nweeks)]
qdl_per_w = [[0] * 1440 for _ in range(nweeks)]
noon_val = [[0] for _ in range(nweeks)]
qdl_aprox = [[0] * 24 for _ in range(nweeks)]
fit_qdl = [[0] * 24 for _ in range(nweeks)]
season_sample = []

for i in range(nweeks):

    fw = idx_daily[i] + timedelta(days=lenght_window)
    
    iw = idx_daily[i]
    
    idx_daily = idx_daily + timedelta(days=10)
  
    tmp_inicio = str(iw)[0:10]
    
    tmp_final= str(fw)[0:10]

    data_per_window = data[tmp_inicio:tmp_final]   

    # Proceed only if the current month data is not all NaN
    if not np.all(np.isnan(data_per_window)):
   
   # Create a date range for the days in the current month
        tmp_daily = pd.date_range(start=pd.Timestamp(tmp_inicio), 
                                    end=pd.Timestamp(tmp_final) + pd.DateOffset(hours=23, minutes=59), freq='D')
         
    # Calculate IQR picks for the current month data
        iqr_picks = max_IQR(data_per_window, 60, 24)
        
        iqr_picks = np.array(iqr_picks)

        valid_days = iqr_picks[~np.isnan(iqr_picks)]
         
        # Only proceed if there are more than 9 valid days
        if len(valid_days) > 6:
            qdl_list_per_w[i] = get_qd_dd(iqr_picks, tmp_daily, 'qdl', 4)
            
            qdl = [[0] * 1440 for _ in range(4)]  # Initialize the list for quiet days
            
            baseline = []    
            
            for j in range(4):
                qd = str(qdl_list_per_w[i][j])[:10]  # Get the date as string yyyy-mm-dd
                
                qd_arr = data_per_window.loc[qd]
                
                qdl[j] = qd_arr.reset_index(drop=True)
                    
                qd_2h = qdl[j][:240]  # Get the first 2 hours (assuming 1 minute resolution)

                qdl[j] = qdl[j] - np.nanmedian(qd_2h)  # Adjust the data by subtracting the median
                
                baseline.append(np.nanmedian(qd_2h))
                
            qdl_per_w[i] = np.nanmedian(qdl, axis=0)  # Calculate the median for the month
            
            noon_val[i] = np.nanmedian(qdl_per_w[i][300:480])

        else:   
            noon_val[i] = np.nan
x = np.linspace(1, len(noon_val)-1, len(noon_val))            

plt.plot(x, noon_val, 'o')

#plt.plot(data.index, data)

#plt.legend()


noon_val = np.array(noon_val)
noon_val = fill_gap(noon_val)
############################################################################### 
############################################################################### 

    #print(season_sample)


xdata = np.linspace(0, len(noon_val)-1, len(noon_val))
    #print(season_data)
    #plt.plot(xdata, noon_val, 'o')
    

    # Define a Fit object for this model and data

fit_season = fit_data(noon_val)

plt.plot(fit_season)

plt.show()

xmin = np.linspace(0, ndata-1, ndata)

#agregar un proceso extra para eliminar artefacto    

interpol = splrep(xdata, fit_season,k=3,s=5)
        
# Evaluar la interpolación en puntos específicos
time_axis = np.linspace(min(xdata), max(xdata), ndata)
season_aprox = splev(time_axis, interpol)



baseline, trend, DD = base_line(data, idx, idx_daily)

H_det = data - baseline  

plt.figure(figsize=(12,6))
plt.plot(H_det)
plt.plot(data.index, season_aprox)
plt.show()
#plt.savefig("/home/isaac/tools/test/test_isaac/"+st+'_trend.png')
