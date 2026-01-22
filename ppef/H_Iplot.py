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

station_pairs = [('teo', 'jai'), ('sjg', 'bmt'), ('gui', 'kak'), ('tam', 'hon')]

# Panel 1: ASYH (already defined separately)
# We'll create a separate ASYH panel if needed, or you can keep it as panel 0
fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=False)
# Panel 0: ASYH (from first station)
df_asy = pd.read_csv(f'{path}{st_sect[0]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
ASYH = df_asy.iloc[:, 1]
axes[0].plot(time_m, ASYH, color='black', linewidth=2)
axes[0].set_xlim(time_m[0], time_m[-1])
axes[0].set_ylabel('ASYH [nT]', fontsize=20)
axes[0].grid(True, alpha=0.5)
axes[0].tick_params(labelsize=20)
axes[0].xaxis.set_visible(False)
# Add shaded regions to ASYH panel
axes[0].axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{idate} 15:10:00'), 
              alpha=0.3, color='gray')
axes[0].axvspan(pd.Timestamp(f'{idate} 16:10:00'), pd.Timestamp(f'{idate} 18:00:00'), 
              alpha=0.3, color='gray')
axes[0].axvspan(pd.Timestamp(f'{fdate} 14:10:00'), pd.Timestamp(f'{fdate} 16:10:00'), 
              alpha=0.3, color='gray')
axes[0].axvspan(pd.Timestamp(f'{idate} 23:00:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
              alpha=0.3, color='gray')      
      #print(f'Ob: {st}, UT: {time_h[0]}, UTC: {mlt}')      

mlon = []
df_all = []
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
      mlon.append(lon)
      mlt = []
      
      index=[0,-1]
      for j in range(2):
            tmp_mlt = apex_out.mlon2mlt(int(float(info[9])), time_m[index[j]])
            duration = datetime.timedelta(hours=tmp_mlt)
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
      
      mlt_series = mlt_series[0:-1]

      mlt_hours = (mlt_series.hour
                  + mlt_series.minute / 60
                  + (mlt_series.day - mlt_series[0].day) * 24)       
      
      UT_hours = (time_m.hour
                  + time_m.minute / 60
                  + (time_m.day - time_m[0].day) * 24)      
            
      UT_hours = UT_hours
      mlt_hours = mlt_hours
      
      UT_bin = (UT_hours * 60).astype(int)      # minutos UT
      MLT_bin = (np.array(mlt_hours) * 60).astype(int)        
      
      df_station = pd.DataFrame({
      'UT_min': UT_bin,
      'MLT_min': MLT_bin,
      'H_I': H_I,
      'station': st_sect[st]
      })
      
      print(df_station)
      
      df_all.append(df_station)

df_all = pd.concat(df_all, ignore_index=True)
bin_min = 10  # minutos


df_all['UT_bin']  = (df_all['UT_min']  // bin_min) * bin_min
df_all['MLT_bin'] = (df_all['MLT_min'] // bin_min) * bin_min

Z = df_all.pivot_table(
    values='H_I',
    index='MLT_bin',
    columns='UT_bin',
    aggfunc='mean'
)

UT_grid = Z.columns.values / 60     # a horas
MLT_grid = Z.index.values / 60

UTg, MLTg = np.meshgrid(UT_grid, MLT_grid)
Zvals = np.ma.masked_invalid(Z.values)
      
levels = np.arange(-150, 150, 10)
from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vmin=-150, vcenter=0, vmax=150)
pcm = axes[-1].pcolormesh(
    UTg,
    MLTg,
    Zvals,
    cmap='seismic',
    norm=norm,
    shading='nearest',
    hatch='O'
)

axes[-1].set_ylabel('MLT', fontsize=20)
axes[-1].set_ylim(0, 24)
axes[-1].set_yticks([6, 12, 18])
axes[-1].set_yticklabels(['06', '12', '18'], fontsize=20)
axes[-1].tick_params(labelsize=20)

# MLT reference lines
for h in [6, 12, 18]:
    axes[-1].axhline(h, color='k', linestyle='--', linewidth=0.8)

# Manual colorbar axes: [left, bottom, width, height]
cax = fig.add_axes([0.15, 0.04, 0.70, 0.015])

cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
cbar.set_label(r'$H_I$ [nT]', fontsize=20)
cbar.ax.tick_params(labelsize=20)
    
xticks_hours = np.arange(0, 49, 6)

xtick_labels = [
    time_m[int(i / 48 * (len(time_m) - 1))].strftime('%H:%M')
    for i in xticks_hours
]

axes[-1].set_xticks(xticks_hours)
axes[-1].set_xticklabels(xtick_labels, fontsize=20)
axes[-1].set_xlabel('Universal Time', fontsize=20)




# Loop through station pairs (panels 1-4)
for pair_idx, (station1, station2) in enumerate(station_pairs):
    ax = axes[pair_idx + 1]  # +1 because panel 0 is ASYH
    
    # Load and plot first station
    df1 = pd.read_csv(f'{path}{station1}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I1 = df1.iloc[:, 0]
    ax.plot(time_m, H_I1, color='red', linewidth=2, label=f'{station1.upper()}')
    
    # Load and plot second station
    df2 = pd.read_csv(f'{path}{station2}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I2 = df2.iloc[:, 0]
    ax.plot(time_m, H_I2, color='blue', linewidth=2, label=f'{station2.upper()}')
    
    # Set y-limits and labels
    ax.set_xlim(time_m[0], time_m[-1])
    ax.set_ylim(-160, 160)
    ax.set_ylabel(r'$\Delta \mathrm{mlon}$' '\n' r'$H_I$ [nT]', fontsize=20)
    ax.grid(True, alpha=0.5)
    ax.tick_params(labelsize=20)
    ax.legend(loc='upper right', fontsize=20)
    ax.xaxis.set_visible(False)
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





plt.subplots_adjust(hspace=0.1, bottom=0.1, top=0.95, right=0.95, left=0.1)
#plt.tight_layout()
plt.savefig(f'/home/isaac/longitudinal_studio/fig/asyh_{idate}_{fdate}.png', dpi=300)
plt.close()