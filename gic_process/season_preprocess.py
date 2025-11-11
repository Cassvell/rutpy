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
#idate = sys.argv[1]# "formato(yyyymmdd)"
#fdate = sys.argv[2]

def window_27(idate, fdate):
    ndays = calculate_days_difference(idate, fdate)

    idate = datetime.strptime(idate + ' 00:00:00', '%Y%m%d %H:%M:%S')
    fdate = datetime.strptime(fdate + ' 23:59:00', '%Y%m%d %H:%M:%S')

    nwindows = int(ndays/27)
    iwindows = []
    fwindows = []
    for w in range(nwindows):
        window_start = idate + timedelta(days=w * 27)
        window_end = window_start + timedelta(days=26, hours=23, minutes=59)
        
        if window_end > fdate:
            window_end = fdate
        
        doi_start = window_start.timetuple().tm_yday
        doi_end = window_end.timetuple().tm_yday
        year_start = window_start.year
        year_end = window_end.year
        
        #print(f'Window {w+1}: {year_start:04d}-{doi_start:03d} to {year_end:04d}-{doi_end:03d}')
        tmp_idate = f"{year_start:04d}-{doi_start}"
        tmp_fdate = f"{year_end:04d}-{doi_end}"
        
        
        
        
        iwindows.append(tmp_idate)
        fwindows.append(tmp_fdate)
    return iwindows, fwindows
    #print(f'\n Window {w+1}: {int(tmp_idate)} to {int(tmp_fdate)} \n')
    #subprocess.run(['python', 'gic_process.py', tmp_idate, tmp_fdate])
    
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
