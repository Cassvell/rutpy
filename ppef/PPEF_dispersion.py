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
        
sys.exit('end')



for vt, color in zip(vertical_times, colors):
    axes[0].axvline(x=pd.Timestamp(f'{idate} {vt}'), color=color, 
                    linestyle='--', alpha=0.7, linewidth=1.5)

for i in range(len(st_sect)):
    if i < 4:  # Only plot first 4 stations (panels 2-5)
        print(f'Observatorio: {st_sect[i]}')
        df = pd.read_csv(f'{path}{st_sect[i]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
        H_I = df.iloc[:, 0]
        ASYH = df.iloc[:, 1]
        
        #line1 = ax1.plot(time, ASYH, color='black', linewidth=3, label='ASYH')
        
        # Calculate and plot dASYH on right y-axis (ax1_right)
        dASYH = np.gradient(ASYH, np.linspace(0, len(time), len(time)))
        #line2 = ax1_right.plot(time, np.abs(dASYH), color='cornflowerblue', linewidth=1, linestyle='-', alpha=0.3)
        
        #axes[0].set_title('ASYH Index')
        axes[0].plot(time, ASYH, color='black', linewidth=2, label='ASYH')
        axes[0].set_ylabel('ASYH [nT]', fontsize=20)
        #ax1_right.set_ylabel(r'$ \Delta_t | ASYH|$ [nT/min]', fontsize=20, color='black')
        axes[0].grid(True, alpha=0.5)
        axes[0].tick_params(labelsize=20)        
        
        axes[0].axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{idate} 15:10:00'), 
              alpha=0.3, color='gray')
        
        axes[0].axvspan(pd.Timestamp(f'{idate} 16:10:00'), pd.Timestamp(f'{idate} 18:00:00'), 
              alpha=0.3, color='gray')        
        
        axes[0].axvspan(pd.Timestamp(f'{fdate} 14:10:00'), pd.Timestamp(f'{fdate} 16:10:00'), 
              alpha=0.3, color='gray')

        axes[0].axvspan(pd.Timestamp(f'{idate} 23:00:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
              alpha=0.3, color='gray')        
        
        axes[i+1].plot(time, H_I, color='red')
        axes[i+1].set_ylim(-160,160)
        #axes[i+1].set_title(f'Station: {st_sect[i]}')
        axes[i+1].set_ylabel(rf' {st_sect[i].upper()} $H_I$ [nT]', fontsize=20)
        axes[i+1].grid(True, alpha=0.5)
        axes[i+1].tick_params(labelsize=20)
        for vt, color in zip(vertical_times, colors):
            axes[i+1].axvline(x=pd.Timestamp(f'{idate} {vt}'), color=color, 
                            linestyle='--', alpha=0.7, linewidth=1.5)
        axes[i+1].axvspan(pd.Timestamp(f'{idate} 13:07:00'), pd.Timestamp(f'{idate} 15:10:00'), 
              alpha=0.3, color='gray')
        
        axes[i+1].axvspan(pd.Timestamp(f'{idate} 16:10:00'), pd.Timestamp(f'{idate} 18:00:00'), 
              alpha=0.3, color='gray')        
        
        axes[i+1].axvspan(pd.Timestamp(f'{fdate} 14:10:00'), pd.Timestamp(f'{fdate} 16:10:00'), 
              alpha=0.3, color='gray')

        axes[i+1].axvspan(pd.Timestamp(f'{idate} 23:00:00'), pd.Timestamp(f'{fdate} 00:30:00'), 
              alpha=0.3, color='gray')

# Set x-axis label only on bottom panel
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(' %m/%d %H h'))
axes[-1].set_xlabel('Universal Time', fontsize=20)
#axes[-1].tick_params(labelsize=16)    
# Set x-limits
for ax in axes:
    ax.set_xlim(time[0], time[-1])

plt.suptitle(f'2015/03/17 - 2015/03/18', fontsize=24, y=0.98)
plt.tight_layout()
plt.savefig(f'/home/isaac/longitudinal_studio/fig/asyh_{idate}_{fdate}.png', dpi=300)
plt.show()

sys.exit('end')

for i in range(len(st_sect)):
    print(f'Observatorio: {st_sect[i]}')
    df = pd.read_csv(f'{path}{st_sect[i]}_{idate}_{fdate}.dat', header=None, sep='\\s+')
    
    
    dp2 = df.iloc[:, 2]
    dp2_2 = df.iloc[:,3]
    
    dp2_picks = []
    resample = 15
    num_windows = 2880 // resample

    dp2_picks_2 = []
    num_windows2 = len(dp2_2) // resample
    
    #for l in range(num_windows2):
    #    start_idx = l * resample
    #    end_idx = (l + 1) * resample
    #    tmp = np.nanmax(np.abs(dp2[start_idx:end_idx]))        
    #    dp2_picks_2.append(tmp)   

    nbins = int(len(dp2))
    dp2 = np.array(dp2)  
    dp2_2 = np.array(dp2_2)  
    
    # Clean data
    dp2 = dp2[~np.isnan(dp2)]
    #dp2 = np.unique(dp2)
    
    dp2_2 = dp2_2[~np.isnan(dp2_2)]
    #dp2_2 = np.unique(dp2_2)    
    
    # Histogram and Gaussian fit
    frequencies, bin_edges = np.histogram(dp2, bins=nbins*4, density=True)
    
    frequencies_2, bin_edges_2 = np.histogram(dp2_2, bins=nbins*4, density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    
    bin_centers_2 = (bin_edges_2[:-1] + bin_edges_2[1:]) / 2 
    
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, bin_centers, frequencies, 
                        p0=[1, np.mean(dp2), np.std(dp2)])
    
    popt2, pcov2 = curve_fit(gaussian, bin_centers_2, frequencies_2, 
                        p0=[1, np.mean(dp2_2), np.std(dp2_2)]) 
  
    x_fit = np.linspace(min(bin_edges), max(bin_edges), 500)
    x_fit2 = np.linspace(min(bin_edges_2), max(bin_edges_2), 500)
    fig = plt.figure(figsize=(12, 10))
    station_name = st_sect[i].upper()
    fig.suptitle(f"{station_name}: {idate} to {fdate}", fontsize=24)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0, 0])  # Top lef
    ax2 = plt.subplot(gs[0, 1])  # Top right
    ax3 = plt.subplot(gs[1, :])  # Bottom row (spans both columns)
    ax1.hist(dp2, density=True, bins=int(len(dp2) / 15), color='navy', 
			histtype='stepfilled', alpha=0.4, label=r'$\mathrm{H_{PPEF}}$ distribution')
    ax1.hist(dp2_2, density=True, bins=int(len(dp2_2) / 15), color='orange', 
			histtype='stepfilled', alpha=0.5, label=r'$\mathrm{H_{PPEF2}}$ distribution')    
    ax1.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, 
			label=f'Gaussian fit: μ={popt[1]:.2f}, σ={popt[2]:.2f}')
    ax1.plot(x_fit2, gaussian(x_fit2, *popt2), 'r--', linewidth=2, 
			label=f'Gaussian fit 2: μ={popt2[1]:.2f}, σ={popt2[2]:.2f}')    
    ax1.set_xlim(-75, 75)
    ax1.set_ylim(0, 0.1)
    ax1.set_ylabel('Probability Density', fontsize=20)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)

    #fig.set_title(f"{station_name}: {idate} to {fdate}")

	# --- Top right: CDF of RESAMPLED data (dp2_picks) ---
    sorted_data = np.sort(np.abs(dp2))
    sorted_data_2 = np.sort(np.abs(dp2_2))

    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    cdf2 = np.arange(1, len(sorted_data_2)+1) / len(sorted_data_2)

	# Plot empirical CDF of resampled data
    ax2.plot(sorted_data, cdf, 'b-', label=r'$max[|H_{PPEF}|]$ CDF', linewidth=3)
    ax2.plot(sorted_data_2, cdf2, color='orange', label=r'$max[|H_{PPEF2}|]$ CDF', linewidth=3)

    print(f'max HPPEF: {max(dp2_2)}')
    print(f'min HPPEF: {min(dp2_2)}') 
    params = halfnorm.fit(sorted_data)
    mean, std = params

    percentile = 0.9
    idx = np.searchsorted(cdf, percentile)
    idx_2 = np.searchsorted(cdf2, percentile)

    if idx < len(sorted_data):
    # Add vertical line at the 95% value for resampled data
        value_95 = sorted_data[idx]
        ax2.axvline(x=value_95, color='blue', linestyle='--', alpha=1, linewidth=1.5, 
                label=f'Threshold = {value_95:.2f} nT')

    if idx_2 < len(sorted_data_2):    
        value_95_2 = sorted_data_2[idx_2]
        ax2.axvline(x=value_95_2, color='orange', linestyle='--', alpha=1, linewidth=1.5, 
                label=f'Threshold (TW) = {value_95_2:.2f} nT')

    print(f'Threshold resampled: {value_95}, Threshold in TW: {value_95_2}')        

    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$\mathrm{H_{PPEF}}$ distribution [nT]", fontsize=20)
    ax2.set_ylabel('Cumulative Probability', fontsize=16)
    ax2.legend(loc='lower right', fontsize=14)
    ax2.grid(True, alpha=0.3)

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

    plt.tight_layout()
    plt.savefig(f'{path2}{st_sect[i]}_{idate}_{fdate}.V1.png', dpi=300)
    plt.close()