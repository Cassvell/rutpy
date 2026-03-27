import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from gicdproc import df_gic_pp, df_gic_processed
from scipy.fft import fft, fftfreq
from scipy import signal
import sys
dir_path = '/home/isaac/datos/gics_obs/'

idate = sys.argv[1]
fdate = sys.argv[2]

stations = ['LAV', 'QRO', 'RMY', 'MZT']

def psd(data):
    data = data.flatten()
    ndata = len(data)
    #ESTABLECER FUNCION VENTANA
    fs = 1/60
    
    fny = 1/(2*fs)
    
    f, psd = signal.welch(data, fs, window='hann', nperseg=1024, detrend='constant', scaling='density')
    
    # PSD para frecuencias positivas (usando **2 en lugar de ^)
    
    
    return f, psd, fny

fig, axes = plt.subplots(2, 2, figsize=(14, 14 ))
axes = axes.flatten()

for idx, st in enumerate(stations):
        
    gic_data = df_gic_processed(idate, fdate, f'{dir_path}processed/', st)
    

    gic_prep = df_gic_pp(idate, fdate, f'{dir_path}', st)

    gic_pp =np.array(gic_prep.iloc[:,1])
    gic_processed = np.array(gic_data)
    
    gic_processed_cleaned = gic_processed[~np.isnan(gic_processed)]
    gic_pp_cleaned = gic_pp[~np.isnan(gic_pp)]
    
    fk, pwd_bef, fny = psd(gic_pp_cleaned)
    
    fk, pwd_aft, fny = psd(gic_processed_cleaned)
    
    axes[idx].semilogy(fk, pwd_bef, 'b-', markersize=4, alpha=0.6, label='PSD pre')
    axes[idx].semilogy(fk, pwd_aft, 'r-', markersize=4, alpha=0.8, label='PSD post')

    # Set x-axis to log scale
    axes[idx].set_xscale('log')

    # Define the periods we want to mark
    periods_hours = [48, 24, 12, 6, 4, 2, 1, 0.25, 0.1]
    periods_seconds = [p * 3600 for p in periods_hours]
    freq_positions = [1 / p for p in periods_seconds]

    # Add vertical lines at these periods
    for freq, period in zip(freq_positions, periods_hours):
        axes[idx].axvline(x=freq, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        # Add text label
        axes[idx].text(freq, axes[idx].get_ylim()[1]*1.0, f'{period}h', 
                    rotation=90, fontsize=12, alpha=0.7, ha='right')

    # Regular formatting
    fn = int(len(fk)/2)
    axes[idx].set_xlim(fk[0],fk[fn])
    axes[idx].tick_params(axis='both', which='major', labelsize=14)
    axes[idx].set_ylabel(f'{st} PSD [A]', fontsize=14)
    axes[idx].set_xlabel('Frequency (Hz)', fontsize=14)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

plt.tight_layout()
plt.savefig('/home/isaac/gics_rv/fig/psd.png', dpi=300)
plt.show()