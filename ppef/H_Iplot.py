import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import calendar
from scipy.stats import norm
import matplotlib.dates as mdates
from modules.obs_info import obs_mlon, obs_mlt, compute_mlt_ts

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

#sector = ['TW1', 'TW2']
#####################################################################################################################
#####################################################################################################################
#Defining magnetic stations/Observatories
path_times = '/home/isaac/longitudinal_studio/'
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

df_times = pd.read_csv(f'{path_times}{idate}_{fdate}.txt', header = None)
point_times = df_times[0].tolist()
target_times = []
for t in point_times:
    tmp = pd.Timestamp(t)
    target_times.append(tmp)
    
#target_times = [pd.Timestamp(f'{idate} 13:57:00'), pd.Timestamp(f'{idate} 16:42:00'), 
#                pd.Timestamp(f'{idate} 19:18:00'), pd.Timestamp(f'{idate} 23:40:00')]

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
dt_minutes = 1 
dASYH_dt = np.gradient(ASYH, dt_minutes)

axes[0].plot(time_m, ASYH, color='darkorange', linewidth=2)

for t in target_times:
    idx = time_m.get_loc(t)    
    #axes[0].text(time_m[idx], 5, time_m[idx].strftime('%H:%M'),color='black',fontsize=18,ha='center',va='bottom')

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
    for t in target_times:
        idx = time_m.get_loc(t)
        #print(f'Time: {t}')
        #print(f'H_I {st_sect[st].upper()} = {H_I[idx]}')

    #sys.exit('end')
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

    
xticks_times = pd.date_range(start=time_m[0], end=time_m[-1], freq='6h')
xticks_hours = ( xticks_times.hour + xticks_times.minute / 60 + (xticks_times.day - time_m[0].day) * 24 )

xtick_labels = [t.strftime('%H:%M') for t in xticks_times]

axes[-1].set_xticks(xticks_hours)
axes[-1].set_xticklabels(xtick_labels, fontsize=20)
axes[-1].set_xlabel('Universal Time', fontsize=20)

# --- Líneas verticales en todos los subplots ---
for i in range(len(axes)):
    for vt in vertical_times:
        ts = pd.to_datetime(f"{idate} {vt}")

        if i < len(axes) - 1:
            # Subplots de series temporales
            axes[i].axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
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

    ASYH = df1.iloc[:, 1]
    # Load and plot second station
    df2 = pd.read_csv(f'{path}{station2}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I2 = df2.iloc[:, 0]
    
    mlon_data = obs_mlon([station1, station2])
    
    mlt1 = compute_mlt_ts(mlon_data[0], time_m)    
    mlt2 = compute_mlt_ts(mlon_data[1], time_m)

        
    #dlon = ((mlon_data[1] - mlon_data[0] + 180) % 360) - 180
    #important hours
        
    ax.plot(time_m, H_I1, color='magenta', linewidth=2, label=rf'{station1.upper()}, $\phi_m=$ {mlon_data[0]}°')
    ax.plot(time_m, H_I2, color='green', linewidth=2, label=rf'{station2.upper()}, $\phi_m=$ {mlon_data[1]}°')
    
    
    yval1 = 0
    yval2 = 0
    for t in target_times:
        idx = time_m.get_loc(t)

        ax.plot(time_m[idx],H_I1[idx],marker='o',markersize=10,
                color='magenta',markeredgecolor='black',zorder=5)
        
        print(f'H_I {station1.upper()} = {H_I1[idx]}')
        ax.plot(time_m[idx],H_I2[idx],marker='o',markersize=10,
                color='green',markeredgecolor='black',zorder=5)
        print(f'H_I {station2.upper()} = {H_I2[idx]}\n')
        if H_I1[idx] > 0 and H_I1[idx] > H_I2[idx]:
            yval1 = 110
            yval2 = -170
        else:
            yval1 = -170 
            yval2 = 110
            
        #ax.text(time_m[idx], yval1, mlt1[idx].strftime('%H:%M'),
        #        color='magenta',fontsize=18,ha='center',va='bottom')
        #ax.text(time_m[idx], yval2,mlt2[idx].strftime('%H:%M'),
        #        color='green',fontsize=18,ha='center',va='bottom')
        
    # Set y-limits and labels
    ax.set_xlim(time_m[0], time_m[-1])
    ax.set_ylim(-175, 150)
    ax.set_ylabel(rf'$H_I$ [nT]', fontsize=20)
    ax.grid(True, alpha=0.5)
    ax.tick_params(labelsize=20)
    ax.legend(loc='upper right', fontsize=20)
    ax.xaxis.set_visible(False)    

# Si las soluciones anteriores no funcionan, puedes crear un diccionario manual
month_names_en = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}


start = time_m[0]
end = time_m[-1]
month_name = month_names_en[start.month]
year = start.strftime("%Y")
day_range = f"{start.day} - {end.day}"
fig.text(0.5, 0.99, f"{month_name} {day_range}, {year}", ha='center', va='top', fontsize=30, fontweight='bold')


plt.subplots_adjust(hspace=0.1, bottom=0.05, top=0.93, right=0.95, left=0.084)
#plt.tight_layout()
plt.savefig(f'/home/isaac/longitudinal_studio/fig/asyh_{idate}_{fdate}.png', dpi=300)
plt.close()