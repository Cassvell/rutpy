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
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
###############################################################################
###############################################################################
#FOURIER FIT
###############################################################################
###############################################################################

from magdata_processing import get_qd_dd, max_IQR, base_line



###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################

imonth = ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', \
          '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01']
#imonth = ['2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', '2024-01-01', '2024-02-01', \
#           '2024-03-01', '2024-04-01', '2024-05-01', '202-06-01']


fmonth = ['2020-01-31', '2020-02-28', '2020-03-31', '2020-04-30', '2020-05-31', '2020-06-30', \
          '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31', '2020-11-30', '2020-12-31']
#fmonth = ['2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31', '2024-01-31', '2024-02-29', \
#          '2024-03-31', '2024-04-30', '2024-05-31', '2024-06-30']

#filenames = []
path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 
df = pd.read_excel(path_qdl+'qdl.ods', engine='odf', sheet_name=0)

def nan_array(size):
    return np.full(int(size), np.nan)
def findMiddle(arr):
    l = len(arr)
    if l % 2 == 0:
        # Even length
        return int((arr[l//2 - 1] + arr[l//2]) // 2)
    else:
        # Odd length
        return arr[l//2]

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
baseline, trend, DD = base_line(data, idx, idx_daily)

def seasonal_trend(data, ndata, idx):
###############################################################################
###############################################################################

    def fill_gap(data):
        def nan_helper(y):    
            return np.isnan(y), lambda z: z.nonzero()[0]   
        
        nans, x = nan_helper(data)    
        
        valid_data_indices = x(~nans)
        
        valid_data_values = data[valid_data_indices]
        
        cubic_interp = interp1d(valid_data_indices, valid_data_values, kind='cubic', fill_value="extrapolate")
        
        data[nans] = cubic_interp(x(nans))
        
        return data

    #GET LOCAL QUIET DAYS per month!
    qdl_list_per_month = [[''] * 5 for _ in range(12)]
    qdl_per_month = [[0] * 1440 for _ in range(12)]
    noon_val = [[0] for _ in range(12)]
    qdl_aprox = [[0] * 24 for _ in range(12)]
    fit_qdl = [[0] * 24 for _ in range(12)]
    season_sample = []
    dpm = []
    diurnal_aprox = [[] for _ in range(12)]
    H_dm = []
    
    for i in range(12):
        # Extract data for the current month
        data_per_month = data[imonth[i]:fmonth[i]]

        # Calculate the number of days in the month
        days_per_month = len(data_per_month) // 1440  # Assuming data is minute-wise
        
        days_per_month = int(days_per_month)
        
        dpm.append(days_per_month)
        
        # Proceed only if the current month data is not all NaN
        if not np.all(np.isnan(data_per_month)):

            # Create a date range for the days in the current month
            tmp_daily = pd.date_range(start=pd.Timestamp(imonth[i]), 
                                    end=pd.Timestamp(fmonth[i]) + pd.DateOffset(hours=23, minutes=59), freq='D')
            
            tmp_idx = pd.date_range(start=pd.Timestamp(imonth[i]), 
                                    end=pd.Timestamp(fmonth[i]) + pd.DateOffset(hours=23, minutes=59), freq='T')
            
            # Calculate IQR picks for the current month data
            iqr_picks = max_IQR(data_per_month, 60, 24)

            iqr_picks = np.array(iqr_picks)

            # Filter out NaN values to get valid days
            valid_days = iqr_picks[~np.isnan(iqr_picks)]
            
            # Only proceed if there are more than 9 valid days
            if len(valid_days) > 9:
                
                qdl_list_per_month[i] = get_qd_dd(iqr_picks, tmp_daily, 'qdl', 5)
                
                qdl = [[0] * 1440 for _ in range(5)]  # Initialize the list for quiet days
                
                for j in range(5):
                    
                    qd = str(qdl_list_per_month[i][j])[:10]  # Get the date as string yyyy-mm-dd

                    # Extract the data for the quiet day
                    qd_arr = data_per_month.loc[qd]
                    qdl[j] = qd_arr.reset_index(drop=True)
                    
                    qd_2h = qdl[j][300:480]  # Get the first 2 hours (assuming 1 minute resolution)

                    qdl[j] = qdl[j] - np.nanmedian(qd_2h)  # Adjust the data by subtracting the median

                qdl_per_month[i] = np.nanmedian(qdl, axis=0)  # Calculate the median for the month                
                
               
                #resample data per hourly
                for j in range(24):
                    tmp = np.nanmedian(qdl_per_month[i][j*60:((j+1)*60)-1])
                    
                    qdl_aprox[i].append(tmp)

                qdl_aprox[i] = np.array(qdl_aprox[i][24:48])
                    
                xdata = np.linspace(0, 23, 24)

                fit = Fit(model_dict, x=xdata, y=qdl_aprox[i])
                fit_result = fit.execute()

                fit_qdl[i] = fit.model(x=xdata, **fit_result.params).y

                interpol = splrep(xdata, fit_qdl[i],k=3,s=5)
                    
                time_axis = np.linspace(min(xdata), max(xdata), 1440)
                
                tmp_aprox = splev(time_axis, interpol)
                
                diurnal_aprox[i] = np.tile(tmp_aprox, days_per_month)
                
                baseline_pm = baseline[0:len(data_per_month)]

                tmp = data_per_month - baseline_pm - diurnal_aprox[i] 
                
                tmp = pd.DataFrame(tmp)
                
                tmp = tmp.set_index(tmp_idx)
                
                H_dm.append(tmp)
                
                noon_val[i] = np.nanmedian(qdl_per_month[i][300:360])
            else:   
                noon_val[i] = np.nan

            tmp_monthly_trend = np.full(len(data_per_month) // 60, np.nan)  # Use np.full for NaN initialization
        
            mid_month = len(tmp_monthly_trend) // 2

            tmp_monthly_trend[mid_month] = noon_val[i]

            season_sample.append(tmp_monthly_trend)

    noon_val = np.array(noon_val)
    
    noon_val = fill_gap(noon_val)

    #season_data = np.concatenate(season_sample)

    # Concatenate the flattened list of DataFrames
    H_minus_dm = pd.concat(H_dm, ignore_index=False)
    
    H_minus_dm = H_minus_dm.reindex(idx)

    xdata = np.linspace(0, len(noon_val)-1, len(noon_val))

    fit_season = fit_data(noon_val)

    #agregar un proceso extra para eliminar artefacto    
    interpol = splrep(xdata, fit_season,k=3,s=5)
        
    # Evaluar la interpolación en puntos específicos
    time_axis = np.linspace(min(xdata), max(xdata), ndata)
    
    season_aprox = splev(time_axis, interpol)
    
    # Convert season_aprox to a Series with the same index as H_minus_dm
    season_aprox_series = pd.Series(season_aprox, index=H_minus_dm.index)

    # Ensure season_aprox_series and H_minus_dm have the same length
    if len(season_aprox_series) != len(H_minus_dm):
        raise ValueError("The length of season_aprox_series and H_minus_dm must be the same.")

    # Subtract season_aprox_series from H_minus_dm
    H_detrended = H_minus_dm.subtract(season_aprox_series, axis=0)

    return H_detrended

H_r = seasonal_trend(data, ndata, idx)

#

plt.figure(figsize=(12,6))
plt.plot(H_r['2023-01-01':'2023-02-21'])
#plt.plot(data.index, season_aprox)
plt.legend()
plt.show()
#plt.savefig("/home/isaac/tools/test/test_isaac/"+st+'_trend.png')


'''
'''