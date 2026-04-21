#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:52:48 2024
@author: isaac
"""
import pandas as pd
import numpy as np

import sys
from modules.window_27 import window_27
from modules.diurnal_variation import diurnal_variation_model
from modules.threshold import threshold
from magdata_processing import base_line
from magnetic_datstruct import get_dataframe
from aux_time_DF import index_gen, convert_date
#from Ffitting import fit_data
import os
from modules.moving_window import hourly_IQR
import matplotlib.pyplot as plt
import h5py
###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
###############################################################################
idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

iwindows, medwindows, fwindows, nwindows= window_27(idate, fdate, 'date')
###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(f'{fdate} 23:59:00'), freq='T')
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')                        
filenames = []
dates = []
st = 'coe'
path = f"/home/isaac/datos/regmex/{st}/raw/"
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    fname = f"{st}{date_name_newf}rK.min"
    filenames.append(fname)


magdata = get_dataframe(filenames, st, 'raw',path, idx, dates, 'regmex')
H = magdata['H']
print(len(H))
for w in range(nwindows):
    print(f'Window {iwindows[w]} to {fwindows[w]}:')
    idx_window = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(fwindows[w]), freq='D')
    idx_min = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(f'{fwindows[w]} 23:59:00'), freq='min')

    window_data = H[w*38880:(w+1)*38880]
    iqr_picks = hourly_IQR(window_data, 60, 0.7)
    threshold_mdata = threshold(iqr_picks, iwindows[w], fwindows[w], st, '2s')
    #print(threshold_mdata)
    baseline_curve = base_line(window_data, 'regmex', st, '2s')
    H_corr = window_data - baseline_curve
    sq_gic, monthly_baseline = diurnal_variation_model(window_data, idx_window, 52, st.lower(), 'experimental', threshold_mdata, 'magdata')
    
    #(data, idx_daily, jump_val, st, qd_method, threshold, data_type)
    #sq_gic, monthly_baseline = diurnal_variation_model(window_data, idx_daily, 10, st.lower(), 'experimental', threshold_gic, 'gic')