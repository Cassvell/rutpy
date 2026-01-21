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
import datetime
import time
import matplotlib.dates as mdates
import apexpy
module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

# Now you can import the module
import magdata_processing 
from threshold import max_IQR
#from magdata_processing import mlt
from night_time import night_time

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

sector = ['TW1', 'TW2']

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
st_sect = ['teo', 'jai',  'sjg', 'hon', 'gui', 'kak', 'bmt', 'tam']
colors_list = ['red', 'green', 'goldenrod', 'purple', 'blue', 'orange', 'darkcyan', 'darkorange']
path = '/home/isaac/datos/pca/'
#period = ['1101-2300 LT', '1801-0600 LT', '2201-1000 LT', '0601-1800 LT']

ndata = 2880
window_len = 240

nwindows = ndata // window_len  # 16 ventanas

time_m = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq='min')
time_h = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:00:00', freq='h')
time_d = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:00:00', freq='D')
apex_out = apexpy.Apex(date=2015.3)
#fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
vertical_times = ['04:30:00', '07:00:00', '22:47:00']
colors = ['green', 'green', 'green']

station_pairs = [('teo', 'jai'), ('sjg', 'hon'), ('gui', 'kak'), ('tam', 'bmt')]

# Panel 1: ASYH (already defined separately)
# We'll create a separate ASYH panel if needed, or you can keep it as panel 0
fig, axes = plt.subplots(9, 1, figsize=(16, 16), sharex=True)
# Panel 0: ASYH (from first station)
df_asy = pd.read_csv(f'{path}{st_sect[0]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
ASYH = df_asy.iloc[:, 1]
axes[0].plot(time_m, ASYH, color='black', linewidth=2, label='ASYH')
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

for st in range(len(st_sect)):
      df = pd.read_csv(f'{path}{st_sect[st]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
      H_I = df.iloc[:, 0]
      if not st_sect[st] == 'teo':
            net = 'intermagnet'
      else:
            net = 'regmex'

      info = night_time(net, st_sect[st])
        
      if info[10] == 'W':
        lon =  - int(float(info[9]))
      elif info[10] == 'E':
        lon = int(float(info[9]))
      mlt = []
      for j in range(len(time_m)):
            tmp = apex_out.mlon2mlt(int(float(info[9])), time_m[j])
            duration = datetime.timedelta(hours=tmp)
            total_minutes_td = duration.total_seconds() / 60
            seconds = total_minutes_td * 60
            time_format = time.strftime("%H:%M:%S", time.gmtime(seconds))
            mlt.append(time_format)
      
      idate2 = time_d[0]
      fdate2 = time_d[1] + datetime.timedelta(days=1)
      if datetime.datetime.strptime(mlt[0], "%H:%M:%S").time() >= datetime.time(12, 0):
            idate2 = time_d[0] - datetime.timedelta(days=1)
            fdate2 = fdate2 - datetime.timedelta(days=1)      
      else:
            idate2 = idate2
            fdate2 = fdate2
      
      start_date_str = f"{idate2.year}-{idate2.month:02d}-{idate2.day:02d} {mlt[0]}"
      end_date_str = f"{fdate2.year}-{fdate2.month:02d}-{fdate2.day:02d} {mlt[-1]}"

      # Crear el date_range
      mlt_series = pd.date_range(start=start_date_str, end=end_date_str, freq='min')
      axes[st+1].plot(mlt_series[0:-1], H_I, color=colors_list[st], linewidth=2, label=f'{st_sect[st]}')
      axes[st+1].grid(True, alpha=0.5)
      axes[st+1].legend()
plt.show()
      
      #print(f'Ob: {st}, UT: {time_h[0]}, UTC: {mlt}')      
sys.exit('end')
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