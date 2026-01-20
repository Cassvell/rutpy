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

# Now you can import the module
import magdata_processing 
from threshold import max_IQR
from magdata_processing import mlt
from night_time import night_time

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

sector = ['TW1', 'TW2']

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
st_sect = ['teo', 'jai',  'sjg', 'hon', 'gui', 'kak', 'bmt', 'tam']
colors = ['red', 'green', 'goldenrod', 'purple', 'blue', 'orange', 'darkcyan', 'darkorange']
path = '/home/isaac/datos/pca/'
#period = ['1101-2300 LT', '1801-0600 LT', '2201-1000 LT', '0601-1800 LT']

ndata = 2880
window_len = 240

nwindows = ndata // window_len  # 16 ventanas

time = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq='min')

#fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
vertical_times = ['04:30:00', '07:00:00', '22:47:00']
colors = ['green', 'green', 'green']

station_pairs = [('teo', 'jai'), ('sjg', 'hon'), ('gui', 'kak'), ('tam', 'bmt')]

# Panel 1: ASYH (already defined separately)
# We'll create a separate ASYH panel if needed, or you can keep it as panel 0
fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
# Panel 0: ASYH (from first station)
df_asy = pd.read_csv(f'{path}{st_sect[0]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
ASYH = df_asy.iloc[:, 1]
axes[0].plot(time, ASYH, color='black', linewidth=2, label='ASYH')
axes[0].set_ylabel('ASYH [nT]', fontsize=18)
axes[0].grid(True, alpha=0.5)
axes[0].tick_params(labelsize=16)
axes[0].legend(loc='upper right')

# Add shaded regions to ASYH panel
axes[0].axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{idate} 15:10:00'), 
              alpha=0.3, color='gray')
axes[0].axvspan(pd.Timestamp(f'{idate} 16:10:00'), pd.Timestamp(f'{idate} 18:00:00'), 
              alpha=0.3, color='gray')
axes[0].axvspan(pd.Timestamp(f'{fdate} 14:10:00'), pd.Timestamp(f'{fdate} 16:10:00'), 
              alpha=0.3, color='gray')
axes[0].axvspan(pd.Timestamp(f'{idate} 23:00:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
              alpha=0.3, color='gray')

# Loop through station pairs (panels 1-4)
for pair_idx, (station1, station2) in enumerate(station_pairs):
    ax = axes[pair_idx + 1]  # +1 because panel 0 is ASYH
    
    # Load and plot first station
    df1 = pd.read_csv(f'{path}{station1}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I1 = df1.iloc[:, 0]
    ax.plot(time, H_I1, color='red', linewidth=2, label=f'{station1.upper()}')
    
    # Load and plot second station
    df2 = pd.read_csv(f'{path}{station2}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I2 = df2.iloc[:, 0]
    ax.plot(time, H_I2, color='blue', linewidth=2, label=f'{station2.upper()}')
    
    # Set y-limits and labels
    ax.set_ylim(-160, 160)
    ax.set_ylabel(f'{station1.upper()}/{station2.upper()}\n$H_I$ [nT]', fontsize=16)
    ax.grid(True, alpha=0.5)
    ax.tick_params(labelsize=14)
    ax.legend(loc='upper right', fontsize=12)
    
    # Add vertical lines
    for vt, color in zip(vertical_times, colors):
        ax.axvline(x=pd.Timestamp(f'{idate} {vt}'), color=color, 
                  linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add shaded regions
    ax.axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{idate} 15:10:00'), 
              alpha=0.3, color='gray')
    ax.axvspan(pd.Timestamp(f'{idate} 16:10:00'), pd.Timestamp(f'{idate} 18:00:00'), 
              alpha=0.3, color='gray')
    ax.axvspan(pd.Timestamp(f'{fdate} 14:10:00'), pd.Timestamp(f'{fdate} 16:10:00'), 
              alpha=0.3, color='gray')
    ax.axvspan(pd.Timestamp(f'{idate} 23:00:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
              alpha=0.3, color='gray')

# Set x-axis label only on bottom panel
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(' %m/%d %H h'))
axes[-1].set_xlabel('Universal Time', fontsize=16)

# Set x-limits
for ax in axes:
    ax.set_xlim(time[0], time[-1])

plt.tight_layout()
plt.savefig(f'/home/isaac/longitudinal_studio/fig/asyh_{idate}_{fdate}V2.png', dpi=300)
plt.close()