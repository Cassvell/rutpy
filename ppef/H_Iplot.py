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
import aacgmv2
module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

# Now you can import the module
import magdata_processing 
from threshold import max_IQR
#from magdata_processing import mlt
from night_time import night_time

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

def obs_mlon(obs):
    
    data = []
    
    for i in obs:
        #print(f'Observatorio: {i.lower()}')
        if i.lower() == 'teo':
            net = 'regmex'
        else:
            net = 'intermagnet'
        
        info = night_time(net, i.lower())
        mlon = float(info[9])     # station magnetic longitude
        hemi = info[10]

        if hemi == 'W':
            mlon = -mlon 
        #print(i.lower(), mlon)
        data.append(mlon)
    return(data)

def obs_mlt(obs_mlon, dt):
    mlt = aacgmv2.convert_mlt(obs_mlon, dt, m2a=False) 
    
    return(mlt)

def compute_mlt_ts(obs_mlon, time_m):
    """
    Returns MLT in hours (0–24) for a station over time_m
    """
    mlt_edges = []

    index=[0,-1]
    for j in range(2):
        tmp_mlt = aacgmv2.convert_mlt(obs_mlon, time_m[j], m2a=False)        
        duration = datetime.timedelta(hours=tmp_mlt.item())
        total_minutes_td = duration.total_seconds() / 60
        seconds = total_minutes_td * 60
        time_format = time.strftime("%H:%M:%S", time.gmtime(seconds))
        mlt_edges.append(time_format)
    
    idate2 = time_d[0]
    fdate2 = time_d[1] + datetime.timedelta(days=1)
    if datetime.datetime.strptime(mlt_edges[0], "%H:%M:%S").time() >= datetime.time(12, 0):
        idate2 = time_d[0] - datetime.timedelta(days=1)
        fdate2 = fdate2 - datetime.timedelta(days=1)      
    else:
        idate2 = idate2
        fdate2 = fdate2

    start_date_str = f"{idate2.year}-{idate2.month:02d}-{idate2.day:02d} {mlt_edges[0]}"
    end_date_str = f"{fdate2.year}-{fdate2.month:02d}-{fdate2.day:02d} {mlt_edges[-1]}"
    # Crear el date_range

    mlt_series = pd.date_range(start=start_date_str, end=end_date_str, freq='min')

    mlt_series = mlt_series[0:-1]
    return mlt_series

#sector = ['TW1', 'TW2']
#####################################################################################################################
#####################################################################################################################
#Defining magnetic stations/Observatories

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
st_sect = ['teo', 'jai',  'sjg', 'hon', 'gui', 'kak', 'bmt', 'tam']
station_pairs = [('teo', 'jai'), ('sjg', 'bmt'), ('gui', 'kak'), ('tam', 'hon')]
colors_list = ['red', 'green', 'goldenrod', 'purple', 'blue', 'orange', 'darkcyan', 'darkorange']
path = '/home/isaac/datos/pca/'
#period = ['1101-2300 LT', '1801-0600 LT', '2201-1000 LT', '0601-1800 LT']
#####################################################################################################################
#####################################################################################################################
#defining Time periods
ndata = 2880
window_len = 240

nwindows = ndata // window_len  # 16 ventanas

time_m = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq='min')
time_h = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:00:00', freq='h')
time_d = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:00:00', freq='D')

target_times = [pd.Timestamp('2015-03-17 13:57:00'), pd.Timestamp('2015-03-17 16:42:00'), 
                pd.Timestamp('2015-03-17 23:40:00')]

mlon_data = obs_mlon(st_sect)
mlt_targets = {st: [] for st in st_sect}

for i, st in enumerate(st_sect):
    # mlon_data[i] es el arreglo de esa estación
    for t in target_times:
        mlt_value = obs_mlt(mlon_data[i], t)  # aquí pasas el arreglo y el tiempo
        mlt_targets[st].append(mlt_value)

#fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
vertical_times = ['04:30:00', '07:00:00', '22:47:00']


# Panel 1: ASYH (already defined separately)
fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=False)
#fig.suptitle('March 17 & 18, 2015', fontsize=26, fontweight='bold')

df_asy = pd.read_csv(f'{path}{st_sect[0]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
ASYH = df_asy.iloc[:, 1]

axes[0].plot(time_m, ASYH, color='darkorange', linewidth=2)

for t in target_times:
    idx = time_m.get_loc(t)    
    axes[0].text(time_m[idx], 5, time_m[idx].strftime('%H:%M'),color='black',fontsize=18,ha='center',va='bottom')

    axes[0].plot(time_m[idx],ASYH[idx],marker='o',markersize=10,color='black',markeredgecolor='black',zorder=5)

axes[0].set_xlim(time_m[0], time_m[-1])
axes[0].set_ylabel('ASYH [nT]', fontsize=20)
axes[0].grid(True, alpha=0.5)
axes[0].tick_params(labelsize=20)
axes[0].xaxis.set_visible(False)

# Add shaded regions to ASYH panel     
      #print(f'Ob: {st}, UT: {time_h[0]}, UTC: {mlt}')      

df_all = []
for st in range(len(st_sect)):
    df = pd.read_csv(f'{path}{st_sect[st]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I = df.iloc[:, 0]    
    mlt_series = compute_mlt_ts(mlon_data[st], time_m)    
    
    mlt_hours = (
    mlt_series.hour
    + mlt_series.minute / 60
    )   
        
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
    
    df_all.append(df_station)

df_all = pd.concat(df_all, ignore_index=True)
bin_min = 15  # minutos


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

axes[-1].set(yticklabels=[])
axes[-1].set_yticks([])
ax2 = axes[-1].twinx()
ax2.set_ylabel('MLT', fontsize=20)
ax2.set_ylim(0, 24)
ax2.set_yticks([6, 12, 18])
ax2.set_ylabel("MLT")
ax2.set_yticklabels(['06', '12', '18'], fontsize=20)
ax2.tick_params(labelsize=20)

# MLT reference lines

for h in [6, 12, 18]:
    axes[-1].axhline(h, color='k', linestyle='--', linewidth=0.8)


# Manual colorbar axes: [left, bottom, width, height]
cax = fig.add_axes([0.075, 0.05, 0.01, 0.135])

cbar = fig.colorbar(pcm, cax=cax, orientation='vertical')
cbar.ax.yaxis.set_label_position("left")
cbar.ax.yaxis.tick_left()
cbar.ax.tick_params(left=False, right=False)
cbar.set_label(r'$H_I$ [nT]', fontsize=20)
cbar.ax.yaxis.set_label_position("left")
cbar.ax.tick_params(labelsize=20)
    
xticks_hours = np.arange(0, 49, 6)

xtick_labels = [
    time_m[int(i / 48 * (len(time_m) - 1))].strftime('%H:%M')
    for i in xticks_hours
]

axes[-1].set_xticks(xticks_hours)
axes[-1].set_xticklabels(xtick_labels, fontsize=20)
axes[-1].set_xlabel('Universal Time', fontsize=20)

# --- Líneas verticales en todos los subplots ---
for i in range(len(axes)):
    for vt in vertical_times:
        ts = pd.to_datetime(f"{idate} {vt}")

        if i < len(axes) - 1:
            # Subplots de series temporales
            axes[i].axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{idate} 15:10:00'), 
              alpha=0.3, color='lightgray')
            axes[i].axvspan(pd.Timestamp(f'{idate} 16:10:00'), pd.Timestamp(f'{idate} 18:00:00'), 
                        alpha=0.3, color='lightgray')
            axes[i].axvspan(pd.Timestamp(f'{fdate} 14:10:00'), pd.Timestamp(f'{fdate} 16:10:00'), 
                        alpha=0.3, color='lightgray')
            axes[i].axvspan(pd.Timestamp(f'{idate} 23:00:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
                        alpha=0.3, color='lightgray') 
            
            axes[i].axvline(x=ts, color='black', linestyle='--', linewidth=1.5)
        else:
            # Último subplot: mapa UT–MLT
            ut_hour = ts.hour + ts.minute / 60.0
            axes[i].axvline(x=ut_hour, color='black', linestyle='--', linewidth=1.5)


# Loop through station pairs (panels 1-4)
for pair_idx, (station1, station2) in enumerate(station_pairs):
    ax = axes[pair_idx + 1]  # +1 because panel 0 is ASYH
    
    # Load and plot first station
    df1 = pd.read_csv(f'{path}{station1}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I1 = df1.iloc[:, 0]

    
    # Load and plot second station
    df2 = pd.read_csv(f'{path}{station2}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I2 = df2.iloc[:, 0]
    
    mlon_data = obs_mlon([station1, station2])
    
    mlt1 = compute_mlt_ts(mlon_data[0], time_m)    
    mlt2 = compute_mlt_ts(mlon_data[1], time_m)

        
    #dlon = ((mlon_data[1] - mlon_data[0] + 180) % 360) - 180
    #important hours
        
    ax.plot(time_m, H_I1, color='magenta', linewidth=2, label=rf'{station1.upper()}, $\phi_m=$ {mlon_data[0]}°')
    ax.plot(time_m, H_I2, color='green', linewidth=2, label=f'{station2.upper()}, $\phi_m=$ {mlon_data[1]}°')
    
    yval1 = 0
    yval2 = 0
    for t in target_times:
        idx = time_m.get_loc(t)

        ax.plot(time_m[idx],H_I1[idx],marker='o',markersize=10,
                color='magenta',markeredgecolor='black',zorder=5)
        ax.plot(time_m[idx],H_I2[idx],marker='o',markersize=10,
                color='green',markeredgecolor='black',zorder=5)
        
        if H_I1[idx] > 0 and H_I1[idx] > H_I2[idx]:
            yval1 = 110
            yval2 = -170
        else:
            yval1 = -170 
            yval2 = 110
            
        ax.text(time_m[idx], yval1, mlt1[idx].strftime('%H:%M'),
                color='magenta',fontsize=18,ha='center',va='bottom')
        ax.text(time_m[idx], yval2,mlt2[idx].strftime('%H:%M'),
                color='green',fontsize=18,ha='center',va='bottom')
        
    # Set y-limits and labels
    ax.set_xlim(time_m[0], time_m[-1])
    ax.set_ylim(-175, 150)
    ax.set_ylabel(rf'$H_I$ [nT]', fontsize=20)
    ax.grid(True, alpha=0.5)
    ax.tick_params(labelsize=20)
    ax.legend(loc='upper right', fontsize=20)
    ax.xaxis.set_visible(False)    

#vertical_times = ['04:30:00', '07:00:00', '22:47:00']
for start, end in [(f'{idate} 02:40:00', f'{idate} 05:10:00'), (f'{idate} 05:50:00', f'{idate} 22:30:00')]: 
    # Convertir a números de matplotlib (float) 
    start_num = mdates.date2num(pd.Timestamp(start)) 
    end_num = mdates.date2num(pd.Timestamp(end)) # Usar transform de datos a coordenadas de figura 
    trans = axes[0].get_xaxis_transform() # transforma x en datos, y en axes 
    start_fig = fig.transFigure.inverted().transform(trans.transform((start_num, 1))) 
    end_fig = fig.transFigure.inverted().transform(trans.transform((end_num, 1))) # Dibujar línea horizontal en coordenadas de figura 
    
    fig.lines.append(plt.Line2D([start_fig[0], end_fig[0]], [0.935, 0.935], 
                                transform=fig.transFigure, color='navy', lw=3, ls='-'))

fig.text(0.19, 0.95, 'SSC', ha='center', va='top', fontsize=18, fontweight='bold')
fig.text(0.34, 0.95, 'MP', ha='center', va='top', fontsize=18, fontweight='bold')

fig.text(0.5, 0.99, 'March 17 & 18, 2015', ha='center', va='top', fontsize=30, fontweight='bold')


plt.subplots_adjust(hspace=0.1, bottom=0.05, top=0.93, right=0.95, left=0.084)
#plt.tight_layout()
plt.savefig(f'/home/isaac/longitudinal_studio/fig/asyh_{idate}_{fdate}.png', dpi=300)
plt.close()