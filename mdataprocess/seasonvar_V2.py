#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:52:48 2024
@author: isaac
"""
import pandas as pd
import numpy as np
#from statistics import mode
#from datetime import datetime
# Ajuste de distribuciones
import sys
#from numpy.linalg import LinAlgError
#from scipy.interpolate import splrep, splev
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter1d
#from scipy.interpolate import NearestNDInterpolator
from magnetic_datstruct import get_dataframe
from scipy.signal import medfilt
from aux_time_DF import index_gen, convert_date
from lowpass_filter import aphase, dcomb
from typical_vall import night_hours, mode_nighttime, typical_value, gaus_center, mode_hourly
from threshold import get_threshold, max_IQR, med_IQR
from plots import plot_GPD, plot_detrend, plot_qdl ,plot_process
#from Ffitting import fit_data
from night_time import night_time
import os
from scipy import fftpack, signal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import h5py
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

ventana_final = fdate - timedelta(days=27)-timedelta(hours=23)-timedelta(minutes=59)
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

#from magdata_processing import get_qd_dd, max_IQR, base_line
net = 'regmex'    
path = f"/home/isaac/datos/{net}/{st}/{st}_raw/"
path_qdl = '/home/isaac/rutpy/mdataprocess/qdl_training/' 
h5_filename = "output_data.h5"
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
    new_name = str(date_name_newf)[0:8]
    #fname = st+'_'+new_name+'.min'
    fname = st+'_'+new_name+'.clean.dat'
    filenames.append(fname)


magdata = get_dataframe(filenames, path, idx, dates, net)
data = magdata['H']

ndata = len(data)
ndays = len(idx_daily)
rcc = int(ndays/27)

lenght_window = 27   
nweeks = 112
daily_sample = 1440
ndays = 10
info = night_time(net, st)
utc = -6#info[11]

ini = 0
fin = 0  

freqs = np.array([0.0, 1.1574e-5, 2.3148e-5, 3.4722e-5,4.6296e-5, \
                    5.787e-5, 6.9444e-5])    

n = daily_sample
N = daily_sample

fs = 1/60
f = fftpack.fftfreq(n, 1.0/fs)
f = np.around(f, decimals = 9)
mask = np.where(f >= 0)
f=f[mask]

fcomb = dcomb(n//2,1,f,freqs) 
#print(fcomb.shape)



def get_qd_dd(data, idx_daily, type_list, n):
    
    daily_var = {'Date': idx_daily, 'VarIndex': data}
    
    local_var = pd.DataFrame(data=daily_var)
    
    local_var = local_var.sort_values(by = "VarIndex", ignore_index=True)
    
    if type_list == 'qdl':
    
        local_var = local_var[0:n]['Date']   
    
    elif type_list == 'I_iqr':
    
        local_var = local_var.sort_values(by = "Date", ignore_index=True)
    
    return local_var


with h5py.File(h5_filename, "w") as h5f:
    for h in range(rcc):
            # Move window by 2 days instead of full window shift
        tmp_inicio = h * (2 * daily_sample)  # Shift by 2 days
        tmp_final = tmp_inicio + (lenght_window * daily_sample) - 1
        group = h5f.create_group(f"rcc/{h}")
        srot = np.arange(1,28,1)
        group.create_dataset("srot", data=srot)
        if tmp_final >= len(data):  # Ensure it does not exceed data length
            break

        rcc_data = data[tmp_inicio:tmp_final + 1]

        if not np.all(np.isnan(rcc_data)):
            tmp_daily = pd.date_range(start=pd.Timestamp(idx[tmp_inicio]), 
                                        end=pd.Timestamp(idx[tmp_final]), freq='D')

            # Calculate IQR picks for the current window
            iqr_picks = max_IQR(rcc_data, 60, 24, method='stddev')    
            iqr_picks = np.array(iqr_picks)

            qd_list = get_qd_dd(iqr_picks, tmp_daily, 'qdl', ndays)
            str_dtype = h5py.string_dtype(encoding="utf-8")  # Allow storing variable-length strings
            qd_list_arr = np.array(qd_list, dtype=str_dtype)  # Convert to numpy array
            qdl = [[0] * 1440 for _ in range(ndays)]
            T = np.array([[0] * 1440 for _ in range(ndays)])
            baseline = []
            
            for i in range(ndays):
                qd = (str(qd_list[i])[0:10])
                
                qd_arr = data[qd]
                
                qdl[i] = qd_arr
                
            # plt.plot(xaxis, qdl[i], label=f'QD{i+1}: {qd}')
                if utc <= 0:
                    ini = int(abs(utc)*60)
                    fin = ini+180    
                    qd_2h = qdl[i].iloc[ini:fin]    
                    baseline_value = np.nanmedian(qd_2h)
                    baseline.append(baseline_value)
                elif utc >= 0:
                    ini = int(1440 - abs(utc)*60)
                    if (ini+180) <= 1440:
                        fin = (ini + 180)
                        qd_2h = qdl[i].iloc[ini:fin]
                        baseline_value = np.nanmedian(qd_2h)
                        baseline.append(baseline_value)       
                    else:
                        fin2 = (ini+180)-1440
                        
                        fin1 = ini + 59
                        qd_2h = qdl[i].iloc[0:fin2]   
                        qd_2h2 = qdl[i].iloc[ini:fin1]
                        baseline_value1 = np.nanmedian(qd_2h)
                        baseline_value2 = np.nanmedian(qd_2h2)
                        baseline_value = (baseline_value1 + baseline_value2)/2
                        baseline.append(baseline_value)
                    
                baseline_value = np.nanmedian(qd_2h)
                baseline.append(baseline_value)
                qdl[i] = qdl[i] - baseline_value
                
                qdl[i] = qdl[i].reset_index()
                qdl[i] = qdl[i].drop(columns=['index'])
                qdl[i] = np.array(qdl[i])
                qdl[i] = qdl[i].reshape(-1)  # Converts (1440, 1) to (1440,)
                
                x = np.arange(len(qdl[i]))
                mask = ~np.isnan(qdl[i])

                # Interpolate NaN values
                qdl[i] = np.interp(x, x[mask], qdl[i][mask])  # Fixed indexing issue         
            # qdl[i] = np.reshape(qdl[i], N)
                Gw = fftpack.fft(qdl[i], axis=0)/np.sqrt(n)
                Gw = Gw[0:n//2]
                
                G_filt = Gw*fcomb.T 
            #  Remove all zero comps in G_filt
                G = G_filt[(~np.isnan(G_filt)) & (G_filt != 0)]                # 1x7
                

                k = np.pi/720
                
                td = np.arange(N).reshape(N,1)          # Nx1
                Td = np.kron(np.ones(7), td)            # Nx7
                

                phi = aphase(G)
                X = 2*abs(G)/np.sqrt(n)                                     # 1x7
                Ag = np.cos(k*np.multiply(Td, np.arange(7)) + phi)          # Nx7            
                ii = np.multiply(Ag,X)                                      # Nx7      
                suma = np.sum(ii, axis=1)                                   # Nx1  
                detrd = signal.detrend(suma)   
                T_tmp = np.median(np.c_[suma, detrd], axis=1)   
                T[i] = T_tmp
                #plt.plot(T_tmp)
                #plt.plot(qdl[i]) 
                #plt.show()     
            #print(T.shape)
                # Combine all DataFrames into a single DataFrame with shape (n, 1440)
            #qdl = pd.Series(qdl)

            qd_average = np.mean(T, axis=0, keepdims=True)     
            #print(qd_average.shape)
           # plt.plot(qd_average.T, color='k', linewidth=3)
            
            #plt.show()
            monthly_baseline = np.array(baseline)
            rcdata = rcc_data - (np.median(monthly_baseline))
            group.create_dataset("timewindowdata", data=rcdata)
            group.create_dataset("T", data=T)
            stat = np.zeros((len(tmp_daily)))

            for k in range(len(stat)):
                for l in qd_list:
                    if tmp_daily[k] == l:
                        stat[k] = 1  # Keep it 1 if any match occurs
                        break  # Exit loop early once a match is found (optimization)
            group.create_dataset("stat", data=stat)


            #plt.plot(rcdata)
            #plt.show()

print(f"Data successfully stored in {h5_filename}")  
 


