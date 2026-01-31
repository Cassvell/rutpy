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
#st = ['lzh', 'bmt', 'tam', 'jai', 'cyg', 'teo', 'hon', 'gui',  'kak', 'sjg']
color = ['red', 'olive', 'seagreen', 'magenta', 'purple', 'orange', 'darkgreen', 'salmon', 'sienna', 'gray']
st_sect = ['jai', 'teo', 'gui', 'kak', 'bmt', 'sjg', 'hon', 'tam']
#st_sect = ['gui', 'jai', 'kak', 'teo', ]
colors = ['red', 'green', 'goldenrod', 'purple', 'blue', 'orange', 'darkcyan', 'darkorange']
path = '/home/isaac/datos/pca/'
#period = ['1101-2300 LT', '1801-0600 LT', '2201-1000 LT', '0601-1800 LT']

ndata = 2880
window_len = 240

nwindows = ndata // window_len  # 16 ventanas

time = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq='min')
time_3h = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq=f'{int(window_len/60)}h')

R_by_st = {}

#fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
vertical_times = ['04:30:00', '07:00:00', '22:47:00']
#colors = ['green', 'green', 'green']
# Panel 1: ASYH (top panel)

# Panel 1: ASYH (top panel) with dual y-axes
#ax1 = axes[0]  # Main axis (left y-axis)
#ax1_right = ax1.twinx()  # Create twin axis for right y-axis

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
# Procesar cada par de estaciones
for panel_idx in range(4):  # 4 paneles
    ax = axes[panel_idx]
    
    # Calcular índices de las estaciones para este panel
    st_idx1 = panel_idx * 2      # 0, 2, 4, 6
    st_idx2 = panel_idx * 2 + 1  # 1, 3, 5, 7
    local_time = [[], []]
    for i in [st_idx1, st_idx2]:
        if i < len(st_sect):  # Verificar que existe la estación
            print(f'Observatorio: {st_sect[i]}')
            if st_sect[i] == 'teo':
                net = 'regmex'
            else:
                net = 'intermagnet'
            
            info = night_time(net, st_sect[i])

            lt = mlt(float(info[5]), info[6])

            utc = lt
            if utc > 0: 
                new_idate = time[0] + pd.Timedelta(hours=utc, minutes=00)
                new_fdate = time[-1] + pd.Timedelta(hours=utc, minutes=00)          
            else:
                new_idate = time[0] - pd.Timedelta(hours=utc, minutes=00)
                new_fdate = time[-1] - pd.Timedelta(hours=utc, minutes=00)
            print(utc)
            tmp_local_time = pd.date_range(start=f'{new_idate} 00:00:00', end=f'{new_fdate} 23:59:00', freq='min')
            
            local_time[*,:] = tmp_local_time
            sys.exit('end')
            
            
            
            df = pd.read_csv(f'{path}{st_sect[i]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
            H_I = df.iloc[:, 0]
            ASYH = df.iloc[:, 1]
            
            tmp_r = []
            for j in range(nwindows):
                start_idx = j * window_len
                end_idx = (j + 1) * window_len
                
                H_I_w = H_I[start_idx:end_idx] 
                ASYH_w = ASYH[start_idx:end_idx]
                
                P_w = np.corrcoef(ASYH_w, H_I_w)
                correlacion = P_w[0, 1]
                tmp_r.append(correlacion)
            
            R_by_st[st_sect[i]] = tmp_r
            
            # Graficar en el panel correspondiente
            ax.plot(time_3h, tmp_r, '-', label=f'R: {st_sect[i].upper()}', 
                   color=colors[i], markersize=4)
    
    # Configurar cada panel
    ax.axhline(y=0.75, color='r', linestyle='--', alpha=0.7)  
    ax.axhline(y=-0.75, color='r', linestyle='--', alpha=0.7)
    #ax.axvspan(pd.Timestamp(f'{idate} 14:07:00'), pd.Timestamp(f'{idate} 19:00:00'), 
    #          alpha=0.3, color='gray')
    
    ax.set_xlim(time_3h[0], time_3h[-1])
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax.set_title(f'{st_sect[st_idx1].upper()} & {st_sect[st_idx2].upper()}', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)  
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('R coeff', fontsize=20)
# Título general para toda la figura
fig.suptitle(rf'$H_I$ vs ASYH. Moving window {int(window_len/60)} h: {idate} to {fdate}', 
             fontsize=24, y=0.98)


plt.tight_layout()
plt.savefig(f'/home/isaac/longitudinal_studio/fig/corr/TScorr_{idate}_{fdate}_{int(window_len/60)}')
plt.show()
    #print(f'Ventana {j+1}: r = {correlacion:.4f}')