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


    
def norm_model(dp2):
    
    nbins = int(len(dp2))
    
    dp2 = np.array(dp2)      
    dp2 = dp2[~np.isnan(dp2)]
        
    # Histogram and Gaussian fit
    frequencies, bin_edges = np.histogram(dp2, bins=int(nbins/10), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, bin_centers, frequencies, 
                        p0=[1, np.mean(dp2), np.std(dp2)])
    
    x_fit = np.linspace(min(bin_edges), max(bin_edges), 500)
    y_fit = gaussian(x_fit, *popt)  
    return x_fit, y_fit, popt


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
        dp2_1 = df.iloc[:, 2].dropna().values
        dp2_2 = df.iloc[:, 3].dropna().values

        # =====================
        # Modelos de distribución
        # =====================
        x1, y1, popt1 = norm_model(dp2_1)
        x2, y2, popt2 = norm_model(dp2_2)

        # =====================
        # Histogramas
        # =====================
        ax.hist(dp2_1,density=True,bins=int(len(dp2_1) / 15),color='navy',histtype='stepfilled',alpha=0.4,
            label=r'$\mathrm{H_{PPEF}}$')

        ax.hist(dp2_2,density=True,bins=int(len(dp2_2) / 15),color='orange',histtype='stepfilled',alpha=0.5,
            label=r'$\mathrm{H_{PPEF2}}$')

        # =====================
        # Ajsute de Gaussianas
        # =====================
        ax.plot(x1, y1, 'r-', lw=2,label=fr'$\mu={popt1[0]:.2f},\ \sigma={popt1[1]:.2f}$')

        ax.plot(x2, y2, 'r--', lw=2,label=fr'$\mu={popt2[0]:.2f},\ \sigma={popt2[1]:.2f}$')

        # =====================
        # 6. Estética
        # =====================
        ax.set_xlim(-75, 75)
        ax.set_ylim(0, 0.1)
        ax.grid(True, alpha=0.3)
        ax.set_title(station.upper(), fontsize=15)
        ax.set_xlabel('Magnitude [nT]', fontsize=14)
        if col_idx == 0:
            ax.set_ylabel('Probability Density', fontsize=14)
        ax.legend(fontsize=11)

    # Etiqueta del par (lado izquierdo)
    #axes[pair_idx, 0].annotate(f'Pair {pair_idx + 1}',xy=(-0.35, 0.5),xycoords='axes fraction',fontsize=14,
    #    rotation=90,va='center')
 

plt.tight_layout()
plt.savefig(f'{path2}HPPEFdist_{idate}_{fdate}.png', dpi=300)
plt.close()

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
        dp2_1 = df.iloc[:, 2].dropna().values
        dp2_2 = df.iloc[:, 3].dropna().values


        
        
        ax.plot(time, dp2_1,color='navy',alpha=0.8,label=r'$\mathrm{H_{PPEF}}$')

        ax.plot(time,dp2_2,color='orange',alpha=0.8,label=r'$\mathrm{H_{PPEF}^\prime}$')

        # =====================
        # 6. Estética
        # =====================
        ax.set_xlim(time[0], time[-1])
        #ax.set_ylim(0, 0.1)
        ax.grid(True, alpha=0.3)
        ax.set_title(station.upper(), fontsize=15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))        
        if col_idx == 1:
            ax.legend(fontsize=14, loc='upper right')
        elif col_idx ==0:
            ax.set_ylabel(r'$H_{PPEF}$ [nT]', fontsize=18)
 
fig.text(0.5, 0.99, 'March 17 & 18, 2015', ha='center', va='top', fontsize=22, fontweight='bold')
fig.text(0.5, 0.03, 'Universal Time', ha='center', va='top', fontsize=18)

plt.subplots_adjust(hspace=0.24, wspace=0.1, bottom=0.06, top=0.93, right=0.98, left=0.07)
plt.savefig(f'{path2}HPPEF_{idate}_{fdate}.png', dpi=300)
plt.close()



'''  
# --- Bottom: Time series ---
ax3.plot(time, dp2, color='navy', label=r'$H_{}$')
ax3.axhline(y=popt[1]+value_95, color='blue', linestyle='--', alpha=0.8, linewidth=1)    
ax3.axhline(y=popt[1]+value_95*(-1), color='blue', linestyle='--', alpha=0.8, linewidth=1)      

ax3.axhline(y=popt2[1]+value_95_2, color='darkorange', linestyle='--', alpha=0.8, linewidth=1)    
ax3.axhline(y=popt2[1]+value_95_2*(-1), color='darkorange', linestyle='--', alpha=0.8, linewidth=1)      

ax3.plot(time, dp2_2, color='orange', label=r'$H_{PPEF2}$')
ax3.set_xlim(time[0], time[-1])
ax3.grid(True, alpha=0.3)  
ax3.legend(fontsize=20)
ax3.set_xlabel('UT', fontsize=20)
ax3.set_ylabel(r'$H_{PPEF}$[nT]', fontsize=20)
'''  