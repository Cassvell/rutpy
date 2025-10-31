import pandas as pd
import numpy as np
import sys
from calc_daysdiff import calculate_days_difference
#from Ffitting import fit_data
import os
from scipy import fftpack, signal
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import h5py

import subprocess



###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

ndays = calculate_days_difference(idate, fdate)

idate = datetime.strptime(idate + ' 00:00:00', '%Y%m%d %H:%M:%S')
fdate = datetime.strptime(fdate + ' 23:59:00', '%Y%m%d %H:%M:%S')

nwindows = int(ndays/27)

for w in range(nwindows):
    window_start = idate + timedelta(days=w * 27)
    window_end = window_start + timedelta(days=26, hours=23, minutes=59)
    
    # Ensure we don't go beyond the final date
    if window_end > fdate:
        window_end = fdate
    
    # Extract date components for printing
    iyear = window_start.year
    imonth = window_start.month
    iday = window_start.day
    
    fyear = window_end.year
    fmonth = window_end.month
    fday = window_end.day
    
    #print(f'Window {w+1}: {iyear:04d}{imonth:02d}{iday:02d} to {fyear:04d}{fmonth:02d}{fday:02d}')
    tmp_idate = f"{iyear:04d}{imonth:02d}{iday:02d}"
    tmp_fdate = f"{fyear:04d}{fmonth:02d}{fday:02d}"
    print(f'Window {w+1}: {int(tmp_idate)} to {int(tmp_fdate)}')
    subprocess.run(['python', 'preprocess.py', tmp_idate, tmp_fdate])
    
###############################################################################
###############################################################################
#CALLING THE DATAFRAME IN FUNCTION OF TIME WINDOW
###############################################################################
###############################################################################
#idx = pd.date_range(start = pd.Timestamp(str(idate)), \
#                        end = pd.Timestamp(str(fdate)), freq='T')
#idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
#                        end = pd.Timestamp(str(fdate)), freq='D')                        
#fw_dates = []
