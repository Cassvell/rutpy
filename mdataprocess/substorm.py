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

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')                        
fw_dates = []
df = pd.read_csv(f'{path}{st[i]}_{idate}_{fdate}.dat', sep='\s+')

dp2 = df.iloc[:, 3]