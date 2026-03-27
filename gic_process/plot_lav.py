#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023
H stations = [Coeneo, Teoloyucan, Iturbide]
@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import  df_dH, df_gic_pp, df_gic_processed
from timeit import default_timer as timer
import sys
import pandas as pd
import os.path
import os
import numpy as np
from datetime import datetime, timedelta
from modules.calc_daysdiff import calculate_days_difference
from modules.corr_offset import detect_offset
from modules.threshold import threshold
from modules.moving_window import hourly_IQR, max_IQR, med_IQR
from modules.diurnal_variation import diurnal_variation_model
start = timer()

if len(sys.argv) < 3:
    sys.exit('Usage: script.py H_start i_date [f_date]')

H_stat = sys.argv[1]
i_date = sys.argv[2]

# Set f_date - use provided value or default to i_date
f_date = sys.argv[3] if len(sys.argv) >= 4 and sys.argv[3] else i_date


    
fyear = int(f_date[0:4])
fmonth = int(f_date[4:6])
fday = int(f_date[6:8])


finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]

stat = ['QRO', 'LAV', 'RMY', 'MZT']
idx1 = pd.date_range(start = pd.Timestamp(i_date+ ' 00:00:00'),  end = pd.Timestamp(f_date + ' 23:59:00'), freq='min')
idx_daily = pd.date_range(start = pd.Timestamp(i_date),  end = pd.Timestamp(f_date ), freq='D')

ndays = calculate_days_difference(i_date, f_date)


path2 = '/home/isaac/datos/gics_obs/processed/'
file = []

gic_lav = df_gic_processed(i_date, f_date, path2, stat[1])
gic_qro = df_gic_processed(i_date, f_date, path2, stat[0])
gic_rmy = df_gic_processed(i_date, f_date, path2, stat[2])
gic_mzt = df_gic_processed(i_date, f_date, path2, stat[3])
print(gic_lav)
#T1 = gicTW_lav['T1']
#T2 = gicTW_lav['T2']
dir_path = f'/home/isaac/datos/dH_{str(H_stat)}/'

fdate = datetime.strptime(f_date, '%Y%m%d')
fdate2 = fdate + timedelta(days=1)
fdate2 = str(fdate2.strftime('%Y%m%d'))


##############################################################################################    
end = timer()

print(end - start)   

