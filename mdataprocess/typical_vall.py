#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:36:23 2024

@author: isaac
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statistics import mode
from scipy.optimize import curve_fit 
import sys
import pandas as pd
from aux_time_DF import index_gen, convert_date
from magnetic_datstruct import get_dataframe
from night_time import night_time
'''

st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
fdate = sys.argv[3]

enddata = fdate+ ' 23:59:00'
idx = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='T')
idx_hr = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='H')    
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')


path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)
'''

def night_hours(data, net, st):
    
    info = night_time(net, st)
    utc = info[11]
    try:
        utc = int(utc)  
    except ValueError:
        utc = float(utc)
    
    ndata = len(data)
    
    ndays = int(ndata/1440)
    
    night_data = ndays*3

    inicio = str(data.index[0])
    
    final = str(data.index[-1])

    tmp_daily = pd.date_range(start = pd.Timestamp(inicio), end = pd.Timestamp(final), freq='D')    

    night_data = []

    for i in range(ndays):
        
        daily_dates = str(tmp_daily[i])[0:10]
        
        tmp = data[daily_dates]
        
        if utc <= 0:
            ini = int(abs(utc)*60)
            fin = ini+180    
            tmp_night_data = tmp[ini:fin]        
            night_data.append(tmp_night_data)
        elif utc >= 0:
            ini = int(1440 - abs(utc)*60)
            if (ini+180) <= 1440:
                fin = (ini + 180)
                tmp_night_data = tmp[ini:fin]        
                night_data.append(tmp_night_data)                
                      
            else:
                fin2 = (ini+180)-1440
                
                fin1 = ini + 59
                
                tmp_night_data1 = tmp[0:fin2]        
                tmp_night_data2 = tmp[ini:fin1]  
                tmp_night_data = pd.concat([tmp_night_data1, tmp_night_data2], axis=0)

                night_data.append(tmp_night_data)                
    
    concat_dat = pd.concat(night_data, axis=0, ignore_index=True)
    
    night_data = np.array(concat_dat)    
    
    return night_data    

def mode_nighttime(data, mv, net, st):
    
    info = night_time(net, st)
    utc = info[11]
    try:
        utc = int(utc)  
    except ValueError:
        utc = float(utc)
    ndata = len(data)
    
    ndays = int(ndata/1440)
    
    night_data = ndays*4
    
    mw_sample = int(ndata/mv) 
    
    ac_mode = []

    night_stacked = []

    inicio = str(data.index[0])
    
    final = str(data.index[-1])

    tmp_daily = pd.date_range(start = pd.Timestamp(inicio), end = pd.Timestamp(final), freq='D')    

    night_data = []

    for i in range(ndays):
        
        daily_dates = str(tmp_daily[i])[0:10]
        
        tmp = data[daily_dates]
        
        if utc <= 0:
            ini = int(abs(utc)*60)
            fin = ini+180   
            tmp_night_data = tmp[ini:fin]
            tmp_night_data = np.array(tmp_night_data)        
            night_data.append(tmp_night_data)
               
        elif utc >= 0:
            ini = int(1440 - abs(utc)*60)
            if (ini+180) <= 1440:
                fin = (ini + 180)
                tmp_night_data = tmp[ini:fin]     
                tmp_night_data = np.array(tmp_night_data)   
                night_data.append(tmp_night_data)                
                       
            else:
                fin2 = (ini+180)-1440
                
                fin1 = ini + 59
                
                tmp_night_data1 = tmp[0:fin2]        
                tmp_night_data2 = tmp[ini:fin1]  
                tmp_night_data1 = np.array(tmp_night_data1)
                tmp_night_data2 = np.array(tmp_night_data2)
                tmp_night_data = np.concatenate([tmp_night_data1, tmp_night_data2], axis=0)
                night_data.append(tmp_night_data)        

    for i in range(ndays):
    
        if i == 0:            
            # For the first time window, use the first time window and the next time window
            tw_mode = night_data[i:(i+2)]
            
            ac_mode[i:(i+2)] += tw_mode
        
        elif i == mw_sample - 1:
            # For the last time window, use the previous time window and the last time window
            tw_mode = night_data[(i-2):(i+1)]
            
            ac_mode[(i-2):(i+1)] += tw_mode
        
        else:
            # For all other time window, use the previous time window, the current time window, 
            # and the next time window
            tw_mode = night_data[(i-1):(i+2)]
            
            ac_mode[(i-1):(i+2)] += tw_mode
            
        #tw_mode = np.array(tw_mode)
        sum_mode = np.nanmean(tw_mode)
        
        night_stacked.append(sum_mode)
    
    return night_stacked


def mode_hourly(data, mw):
    ###############################################################################
    # mode movile window
    ###############################################################################  
    ndata = len(data)
    
    ndays = int(ndata/1440)
    
    mw_sample = int(ndata/mw) 
    
    mode_stacked = []      #moda 
    
    ac_mode = []
    
    mode_sampled = []

    for i in range(mw_sample):
        
        mod =  mode(data[i*mw:(i+1)*mw-1])
        
        mode_sampled.append(mod)

    for i in range(mw_sample):
       
        if i == 0:
            # For the first time window, use the first time window and the next time window
            
            tw_mode = mode_sampled[i:(i+2)]
            
            ac_mode[i:(i+2)] += tw_mode
        
        elif i == mw_sample - 1:
            # For the last time window, use the previous time window and the last time window
            tw_mode = mode_sampled[(i-2):(i+1)]
            
            ac_mode[(i-2):(i+1)] += tw_mode
        
        else:
            # For all other time window, use the previous time window, the current time window, 
            # and the next time window
            tw_mode = mode_sampled[(i-1):(i+2)]
            
            ac_mode[(i-1):(i+2)] += tw_mode
        
        sum_mode = np.nanmean(tw_mode)

        mode_stacked.append(sum_mode)
        
    return mode_stacked


def gaus_center(data):
    
    mw = 18
    
    ndata = len(data)
    
    mw_sample = int(ndata / mw) 
    
    avr = np.nanmean(data)
    
    desv_est = np.nanstd(data)
    
    gauss_center = []
    
    # Define a function for the normal distribution
    def gaussian(x, A, mu, sigma):
    
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    data_nonan = data[~np.isnan(data)]
    
    if len(data_nonan) < 2:  # Skip if there's not enough data
        return [np.nan]
    
    hist, bin_edges = np.histogram(data_nonan, bins=mw_sample - 1, density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    
    initial_guess = [1, avr, desv_est]
    
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=2000)
        
        gauss_cen_tmp = popt[1]
    except RuntimeError:
        gauss_cen_tmp = np.nan
    
    gauss_center.append(gauss_cen_tmp)
    
    return gauss_center

def gaus_center2(data, mw, maxfev=2000):
    
    ndata = len(data)
    
    mw_sample = int(ndata / mw)
    
    avr = np.nanmean(data)
    
    desv_est = np.nanstd(data)
    
    gauss_center = []
    
    # Define a function for the normal distribution
    def gaussian(x, A, mu, sigma):
    
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    for i in range(mw_sample):
    
        tmp_sampleddata = data[i * 180:((i + 1) * 180)]
        
        data_nonan = tmp_sampleddata[~np.isnan(tmp_sampleddata)]
        
        if len(data_nonan) < 2:  # Skip if there's not enough data
        
            gauss_center.append(np.nan)
        
            continue
        
        hist, bin_edges = np.histogram(data_nonan, bins=mw_sample - 1, density=True)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
        
        initial_guess = [1, avr, desv_est]
        
        try:
            popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=maxfev)
        
            gauss_cen_tmp = popt[1]
        except RuntimeError:
        
            gauss_cen_tmp = np.nan
        
        gauss_center.append(gauss_cen_tmp)
    
    return gauss_center


def typical_value(mode, center_gauss, ndata):    
    ###########################################################################################
    #Decide weather the typical value for a day is gonna be either the center of gauss fit or mode
    t_val = []
    
    for i in range(int(ndata)):
        #print(center_gauss[i],mode[i])
        if not center_gauss[i] > mode[i]:
            #print(center_gauss[i],mode[i])
            tmp_val = mode[i]
        
        else:
        
            tmp_val = center_gauss[i]    

        t_val.append(tmp_val)
        #print(t_val)
    return t_val
'''
data = get_dataframe(filenames, path, idx, dates)
#
#tv = typical_value(data)

hourly_mode = mode_movil(data, 60)

daily_gauss = gaus_center(data, 1440)
daily_mode = mode_movil(hourly_mode, 24)
daily_sample = len(data)/1440
typical_dailyval = typical_value(daily_mode, daily_gauss, daily_sample)
plt.plot(typical_dailyval)
plt.show()

'''