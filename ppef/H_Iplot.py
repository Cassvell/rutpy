import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import sys
import os
import datetime
import time
from scipy.interpolate import griddata
import apexpy
module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

# Now you can import the module
import magdata_processing 
from threshold import max_IQR
#from magdata_processing import mlt
from night_time import night_time

def symbol_size(y, size_min=10, size_max=60, step=10, max_abs=150):
    mag = np.abs(y)
    sizes = size_min + (size_max - size_min) * (mag / max_abs)
    return np.round(sizes / step) * step

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]


path = '/home/isaac/datos/pca/'
path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
st_sect = ['teo', 'jai',  'sjg', 'hon', 'gui', 'kak', 'bmt', 'tam']

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
fig, axes = plt.subplots(2, 1, figsize=(16, 16), sharex=False)
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
      mlt_series = mlt_series[0:-1]

      mlt_hours = []
      for t in mlt:
            h, m, s = map(int, t.split(':'))
            mlt_hours.append(h + m/60 + s/3600)
      
      UT_hours = (time_m.hour
                  + time_m.minute / 60
                  + (time_m.day - time_m[0].day) * 24)      
      
      n = min(len(UT_hours), len(mlt_hours), len(H_I))

      UT_hours = UT_hours[:n]
      mlt_hours = mlt_hours[:n]
      H_I = H_I.iloc[:n].values
      
      UT_bin = (UT_hours * 60).astype(int)      # minutos UT
      MLT_bin = (np.array(mlt_hours) * 60).astype(int)        
      
      df_station = pd.DataFrame({
      'UT_min': UT_bin,
      'MLT_min': MLT_bin,
      'H_I': H_I,
      'station': st_sect[st]
      })
      df_all.append(df_station)

df_all = pd.concat(df_all, ignore_index=True)
bin_min = 10  # minutos

df_all['UT_bin']  = (df_all['UT_min']  // bin_min) * bin_min
df_all['MLT_bin'] = (df_all['MLT_min'] // bin_min) * bin_min

Z = df_all.pivot_table(
    values='H_I',
    index='MLT_bin',
    columns='UT_bin',
    aggfunc='mean',
    fill_value=0,
)

UT_grid = Z.columns.values / 60     # a horas
MLT_grid = Z.index.values / 60

UTg, MLTg = np.meshgrid(UT_grid, MLT_grid)
Zvals = np.ma.masked_invalid(Z.values)
      
levels = np.arange(-150, 150, 10)
from matplotlib.colors import TwoSlopeNorm

norm = TwoSlopeNorm(vmin=-150, vcenter=0, vmax=150)
pcm = axes[1].pcolormesh(
    UTg,
    MLTg,
    Zvals,
    cmap='seismic',
    norm=norm,
    shading='nearest',
    hatch='O'
)

axes[1].set_ylabel('Magnetic Local Time [h]', fontsize=18)
axes[1].set_ylim(0, 24)
axes[1].tick_params(labelsize=16)

# MLT reference lines
for h in [6, 12, 18]:
    axes[1].axhline(h, color='k', linestyle='--', linewidth=0.8)

cbar = fig.colorbar(pcm, ax=axes[1], orientation='horizontal',
                    pad=0.08, fraction=0.05)
cbar.set_label(r'$H_I$ [nT]', fontsize=16)
cbar.ax.tick_params(labelsize=14)    
    
xticks_hours = np.arange(0, 49, 6)

xtick_labels = [
    time_m[int(i / 48 * (len(time_m) - 1))].strftime('%d %H:%M')
    for i in xticks_hours
]

axes[1].set_xticks(xticks_hours)
axes[1].set_xticklabels(xtick_labels, fontsize=16)
axes[1].set_xlabel('Universal Time', fontsize=18)

plt.xticks(xticks_hours, xtick_labels)
plt.show()
      
