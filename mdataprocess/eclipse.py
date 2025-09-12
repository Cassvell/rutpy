import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from aux_time_DF import index_gen, convert_date


net = sys.argv[1]
st= sys.argv[2]
idate = sys.argv[3]# "formato(yyyymmdd)"
fdate = sys.argv[4]

year = int(idate[0:4])


enddata = fdate+ ' 23:59:00'
idx = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='min') 
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')
filenames = []
dates = []
path = ''

dfc = []
if net == 'regmex':
    #path = f"/home/isaac/datos/{net}/{st}/{st}20240/" # magnetic data path
    path = f"/home/isaac/datos/{net}/{st}/experiment/" # magnetic data path
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        new_name = str(date_name_newf)[0:8]
        fname2 = f'{path}{st}_{new_name}M.dat'
        
        dfs = pd.read_csv(fname2, sep='\\s+', header = None)
        dfc.append(dfs)

df = pd.concat(dfc, ignore_index=True)
df = df.set_index(idx)
df.replace(9999.9, np.nan, inplace=True)

#plt.plot(df.iloc[:,0])
#plt.ylabel(f'{st.upper()} dH [nT]')
#plt.tight_layout()
#plt.show()


time_array = pd.date_range('00:00', '23:59', freq='1min').strftime('%H:%M')

start_idx = 20 * 60  # 20:00 = 20*60 = 1200 minutes
end_idx = 21 * 60    # 21:00 = 21*60 = 1260 minutes

time_window = time_array[start_idx:end_idx+1]  

plt.figure(figsize=(12, 8))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

ax1.plot(df.iloc[:,0], color='black')
ax1.set_ylabel(f'{st.upper()} dH [nT]')

# Full day plot
for i in range(len(idx_daily)):
    tmp = df.iloc[i*1440:(i+1)*1440, 0]
    if dates[i] == '2024-04-08':
        ax2.plot(time_array, tmp, label=f'{dates[i]}', color='black', linewidth=2.5)
    else:
        ax2.plot(time_array, tmp, label=f'{dates[i]}', alpha=0.7)
ax2.axvspan(time_array[start_idx], time_array[end_idx], alpha=0.2, color='red')
ax2.set_xticks(time_array[::60])
ax2.set_xticklabels(time_array[::60], rotation=45)
ax2.set_ylabel(f'{st.upper()} daily dH [nT]')
ax2.set_title('All daily variations')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 20-21 hour zoom plot
for i in range(len(idx_daily)):
    full_day = df.iloc[i*1440:(i+1)*1440, 0]
    tmp_window = full_day.iloc[start_idx:end_idx+1]
    if dates[i] == '2024-04-08':
        ax3.plot(time_window, tmp_window, color='black', linewidth=2.5, label=f'{dates[i]}', alpha=1, marker='o', markersize=3)

    else:    
        ax3.plot(time_window, tmp_window, label=f'{dates[i]}', alpha=0.7, marker='o', markersize=3)        
ax3.set_xticks(time_window[::10])
ax3.set_xticklabels(time_window[::10], rotation=45)
ax3.set_ylabel(f'{st.upper()} daily dH [nT]')
ax3.set_title('Zoom: 20:00 to 21:00 hour window')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'/home/isaac/MEGA/posgrado/doctorado/semestre5/{st}eclipse.png')
plt.show()



zoom_dates = dates[1:-1]
plt.figure(figsize=(12, 8))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

ax1.plot(df.iloc[1440:-1440,0], color='black')
ax1.set_ylabel(f'{st.upper()} dH [nT]')

# Full day plot
for i in range(len(zoom_dates)):
    tmp = df.iloc[i*1440:(i+1)*1440, 0]
    ax2.plot(time_array, tmp, label=f'{zoom_dates[i]}', alpha=0.7)
ax2.axvspan(time_array[start_idx], time_array[end_idx], alpha=0.2, color='red')
ax2.set_xticks(time_array[::60])
ax2.set_xticklabels(time_array[::60], rotation=45)
ax2.set_ylabel(f'{st.upper()} daily dH [nT]')
ax2.set_title('All daily variations')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 20-21 hour zoom plot
for i in range(len(zoom_dates)):
    full_day = df.iloc[i*1440:(i+1)*1440, 0]
    tmp_window = full_day.iloc[start_idx:end_idx+1]
    ax3.plot(time_window, tmp_window, label=f'{zoom_dates[i]}', alpha=0.7, marker='o', markersize=3)
ax3.set_xticks(time_window[::10])
ax3.set_xticklabels(time_window[::10], rotation=45)
ax3.set_ylabel(f'{st.upper()} daily dH [nT]')
ax3.set_title('Zoom: 20:00 to 21:00 hour window')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()