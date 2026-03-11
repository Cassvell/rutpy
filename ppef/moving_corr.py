import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from H_Iplot import obs_mlon, obs_mlt, compute_mlt_ts

module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

# Now you can import the module
import magdata_processing 
from modules.threshold import max_IQR
from magdata_processing import mlt
from night_time import night_time

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

sector = ['TW1', 'TW2']

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
#st = ['lzh', 'bmt', 'tam', 'jai', 'cyg', 'teo', 'hon', 'gui',  'kak', 'sjg']
color = ['red', 'olive', 'seagreen', 'magenta', 'purple', 'orange', 'darkgreen', 'salmon', 'sienna', 'gray']
st_sect = ['jai', 'teo', 'gui', 'kak', 'bmt', 'sjg', 'hon', 'tam']
station_pairs = [('teo', 'jai'), ('sjg', 'bmt'), ('gui', 'kak'), ('tam', 'hon')]

path = '/home/isaac/datos/pca/'
#period = ['1101-2300 LT', '1801-0600 LT', '2201-1000 LT', '0601-1800 LT']


time_m = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq='min')
ndata = len(time_m)

window_len = 240
time_nh = pd.date_range(start=f'{idate} 00:00:00', end=f'{fdate} 23:59:00', freq=f'{int(window_len/60)}h')

nwindows = ndata // window_len  # 16 ventanas

R_by_st = {}

fig, axes = plt.subplots(4, 1, figsize=(15, 10))
axes = axes.flatten()
# Procesar cada par de estaciones


for pair_idx, (station1, station2) in enumerate(station_pairs):
    ax = axes[pair_idx]  # +1 because panel 0 is ASYH
    
    # Load and plot first station
    df1 = pd.read_csv(f'{path}{station1}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    ASYH = df1.iloc[:, 1]
    H_I1 = df1.iloc[:, 0]

    
    # Load and plot second station
    df2 = pd.read_csv(f'{path}{station2}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    H_I2 = df2.iloc[:, 0]
    
    mlon_data = obs_mlon([station1, station2])
    
    mlt1 = compute_mlt_ts(mlon_data[0], time_m)    
    mlt2 = compute_mlt_ts(mlon_data[1], time_m)

    mlt_nh1 = pd.date_range(start=mlt1.min(), end=mlt1.max(), freq='4h')
    mlt_nh2= pd.date_range(start=mlt2.min(), end=mlt2.max(), freq='4h')
    tmp_r1 = []
    tmp_r2 = []
    for j in range(nwindows):
        start_idx = j * window_len
        end_idx = (j + 1) * window_len
        
        H_I1_w = H_I1[start_idx:end_idx] 
        H_I2_w = H_I2[start_idx:end_idx] 
        ASYH_w = ASYH[start_idx:end_idx]
        
        P_w1 = np.corrcoef(ASYH_w, H_I1_w)
        P_w2 = np.corrcoef(ASYH_w, H_I2_w)
        
        correlacion = P_w1[0, 1]
        correlacion2 = P_w2[0, 1]
        
        tmp_r1.append(correlacion)    
        tmp_r2.append(correlacion2) 
        
    color1 = 'darkgreen'
    color2 = 'magenta'
    ax.plot( mlt_nh1, tmp_r1, '-', label=rf'$\rho_{{{station1.upper()}}}$', color=color1, markersize=4 )
    ax.set_xlim(mlt_nh1[0], mlt_nh1[-1])
    ax.set_ylim(-1, 1)  
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M')) 
    ax.tick_params(axis='both', which='major', labelsize=13)  
    #ax.spines['bottom'].set_color(color1) 
    #ax.tick_params(axis='x', colors=color1)
    
    
    ax2 = ax.twiny() 
    ax2.plot( mlt_nh2, tmp_r2, '-', label=rf'$\rho_{{{station2.upper()}}}$', color=color2, markersize=4 )
    ax2.set_xlim(mlt_nh2[0], mlt_nh2[-1]) 
    ax2.set_ylim(-1, 1)  
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M')) 
    ax2.tick_params(axis='both', which='major', labelsize=13)     
    #ax2.spines['top'].set_color(color2) 
    #ax2.tick_params(axis='x', colors=color2)
    
    
    # Configurar cada panel
    ax.axhline(y=0.75, color='k', linestyle='--', alpha=0.4)  
    ax.axhline(y=-0.75, color='k', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    lines1, labels1 = ax.get_legend_handles_labels() 
    lines2, labels2 = ax2.get_legend_handles_labels() # Combinar 
    lines = lines1 + lines2 
    labels = labels1 + labels2 # Crear una sola leyenda en el panel principal 
    #ax.legend(lines, labels, fontsize=16, loc='center right', frameon=True, framealpha=1)

    ax.grid(True, alpha=0.3)
    ax.set_ylabel(r"$\rho$", fontsize=14)  
    ax.tick_params(axis='y', which='both', labelleft=True)  # mostrar ticks y etiquetas

# Ajustar xlabel según índice 
    if pair_idx in [0, 2]: 
        ax.text(0.0, -0.2, f"{station1.upper()} MLT,", transform=ax.transAxes, fontsize=14,  ha='left', va='top')
        ax.text(0.07, -0.2, rf'$\rho_{{{station1.upper()}}}$', transform=ax.transAxes, fontsize=14, color=color1, ha='left', va='top')
        
        ax2.text(0.0, 1.35, f"{station2.upper()} MLT,", transform=ax2.transAxes, fontsize=14, ha='left', va='top') 
        ax2.text(0.07, 1.35, rf'$\rho_{{{station2.upper()}}}$', transform=ax2.transAxes, fontsize=14, color=color2, ha='left', va='top') 
    else:  
        ax.text(0.96, -0.2, f"{station1.upper()} MLT,", transform=ax.transAxes, fontsize=14, ha='right', va='top')    
        ax.text(1.0, -0.2, rf'$\rho_{{{station1.upper()}}}$', transform=ax.transAxes, fontsize=14, color=color1, ha='right', va='top')
        
        ax2.text(0.96, 1.35, f"{station2.upper()} MLT,", transform=ax2.transAxes, fontsize=14, ha='right', va='top')  
        ax2.text(1.0, 1.35, rf'$\rho_{{{station2.upper()}}}$', transform=ax2.transAxes, fontsize=14, color=color2, ha='right', va='top') 
    
    



    
    
plt.subplots_adjust(hspace=0.75,  bottom=0.06, top=0.94, right=0.98, left=0.065)
for i in range(1, 4): 
    y =  1 - i / 4 * (0.94 - 0.06) / (0.94 - 0.06)
    fig.add_artist(plt.Line2D([0.0, 1], [y, y], transform=fig.transFigure, color='black', lw=1, alpha=0.7))
#fig.suptitle(rf'$H_I$ vs ASYH. Moving window {int(window_len/60)} h: {idate} to {fdate}', 
#             fontsize=24, y=0.98)
plt.savefig(f'/home/isaac/longitudinal_studio/fig/corr/TScorr_{idate}_{fdate}_{int(window_len/60)}')
plt.close()
    #print(f'Ventana {j+1}: r = {correlacion:.4f}')