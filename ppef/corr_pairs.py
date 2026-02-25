import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import os
from scipy.stats import genpareto, kstest #anderson
import kneed as kn
from lmoments3 import distr
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import halfnorm, rayleigh
import matplotlib.dates as mdates

module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

sector = ['TW1', 'TW2']

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'

st_sect = ['teo', 'sjg','gui', 'tam', 'jai', 'bmt',  'kak', 'hon']
station_pairs = [('teo', 'jai'), ('sjg', 'bmt'), ('gui', 'kak'), ('tam', 'hon')]
path = '/home/isaac/datos/pca/'

ndata = 2880
window_len = 240

nwindows = ndata // window_len  # 16 ventanas

time = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq='min')
time_3h = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq=f'{int(window_len/60)}h')
        

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
for pair_idx, (station1, station2) in enumerate(station_pairs):

    for col_idx, station in enumerate([station1, station2]):

        ax = axes[pair_idx, col_idx]

        # =====================
        #Leer archivo
        # =====================
        df = pd.read_csv(f'{path}{station}_{idate}_{fdate}.dat',header=None,sep='\\s+')

        # =====================
        # Series (columnas 2 y 3)
        # =====================
        H_I = df.iloc[:, 2].dropna().values        
        asyH = df.iloc[:, 1].dropna().values
        
        