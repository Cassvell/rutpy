import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from scipy import signal
from scipy.fft import fft, ifft
from gicdproc import pproc, reproc, df_gic, df_dH, df_Kloc, fix_offset, process_station_data, df_sym
from scipy import interpolate

def interpolate_nan(data):
    """Interpolate NaN values in a 1D array"""
    indices = np.arange(len(data))
    mask = ~np.isnan(data)
    if np.sum(mask) > 1:  # Need at least 2 points to interpolate
        f = interpolate.interp1d(indices[mask], data[mask], kind='linear', 
                                fill_value="extrapolate")
        return f(indices)
    else:
        return data  # Return as-is if not enough data

dirpath_sym = '/home/isaac/datos/sym/daily/'

idate = sys.argv[1]
fdate = sys.argv[2]

idx1 = pd.date_range(start = pd.Timestamp(idate), \
                          end = pd.Timestamp(fdate), freq='min')

index = df_sym(idate, fdate, dirpath_sym)

stat = ['LAV', 'QRO', 'RMY', 'MZT']
year = int(idate[0:4])

idx_d = pd.date_range(start = pd.Timestamp(idate + ' 00:00:00'), end= pd.Timestamp(fdate + ' 23:59:59'), freq='D')
ndays = len(idx_d)

file_days = idx_d.strftime('%Y%m%d').tolist()

data_dir = f'/home/isaac/datos/gics_obs/processed/{year}/'
#data_dir2 = f'/home/isaac/datos/regmex/teo/{year}/'
# Initialize station data storage
stat_dir = {s: [] for s in stat}

for s in stat:
    df_tmp = []
    for day in file_days:
        file = os.path.join(data_dir, s, f'gic_{s}_{day}.csv')
        if os.path.isfile(file):
            df = pd.read_csv(file, index_col=0, header=None, parse_dates=True, sep=',')
            
            # CORRECTED: Check if any value in the DataFrame equals 9999.9999
            if (df == 9999.9999).any().any():  # Check if any value in any column is 9999.9999
               # print(f"Found 9999.9999 values in {file} - replacing with NaN")
                df = df.replace(9999.9999, np.nan)
            
            df_tmp.append(df)
        else:
            print(f'File not found: {file} - creating empty DataFrame')
            idx = pd.date_range(
                start=pd.Timestamp(day + ' 00:00:00'),
                end=pd.Timestamp(day + ' 23:59:00'),
                freq='min'
            )
            df = pd.DataFrame(
                np.full((1440, 1), np.nan), index=idx, columns=["1"]
            )
            df_tmp.append(df)
    
    stat_dir[s] = pd.concat(df_tmp, axis=0)
    
'''    
    
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 12))

# Plot data for each station
stations = {
    'ASY': ('black', ax1),
    'LAV': ('blue', ax2),
    'QRO': ('orange', ax3),
    'RMY': ('green', ax4),
    'MZT': ('purple', ax5)
}

for name, (color, ax) in stations.items():
    if name == 'ASY':
        ax.plot(index['DateTime'], index['ASYH'], label='ASYH', color=color, alpha=0.7)
        ax.set_ylabel('ASYM index [nT]')
        ax.set_xlim(index['DateTime'][0], index['DateTime'][-1])
    else:
        ax.plot(stat_dir[name].index, stat_dir[name], label=name, color=color, alpha=0.7)
        ax.set_xlim(stat_dir[name].index[0], stat_dir[name].index[-1])
        ax.legend()
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# Connect click event
plt.tight_layout()
plt.show()
'''
dt = 60.0  # sampling interval in seconds
fs = 1.0 / dt  # sampling frequency in Hz
fny = fs / 2.0  # Nyquist frequency

n = len(index['ASYH'])

asy = np.array(index['ASYH'])
lav = np.array(stat_dir['LAV']).flatten()
rmy = np.array(stat_dir['RMY']).flatten()
mzt = np.array(stat_dir['MZT']).flatten()


lav_interp = interpolate_nan(lav)
rmy_interp = interpolate_nan(rmy)
mzt_interp = interpolate_nan(mzt)


# Or use scipy's built-in function (recommended)
freqs, psd_asy_scipy = signal.welch(asy, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_lav_scipy = signal.welch(lav_interp, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_mzt_scipy = signal.welch(mzt_interp, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_rmy_scipy = signal.welch(rmy_interp, fs=fs, window='hamming', nperseg=n, scaling='density')

periods = 1 / freqs / 3600  # Convert from seconds to hours

# Filter out infinite periods (from frequency = 0)
valid_mask = np.isfinite(periods) & (periods > 0)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: ASYM PSD
axes[0, 0].plot(periods[valid_mask], psd_asy_scipy[valid_mask])
axes[0, 0].set_xlabel('Period [hours]')
axes[0, 0].set_ylabel('PSD ASYM [nT^2/Hz]')
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('ASYM PSD')
axes[0, 0].invert_xaxis()  # Optional: reverse x-axis so longer periods are on the right

# Plot 2: LAV PSD
axes[0, 1].plot(periods[valid_mask], psd_lav_scipy[valid_mask])
axes[0, 1].set_xlabel('Period [hours]')
axes[0, 1].set_ylabel('PSD LAV GIC [A^2/Hz]')
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('LAV GIC PSD')
axes[0, 1].invert_xaxis()

# Plot 3: RMY PSD
axes[1, 0].plot(periods[valid_mask], psd_rmy_scipy[valid_mask])
axes[1, 0].set_xlabel('Period [hours]')
axes[1, 0].set_ylabel('PSD RMY GIC [A^2/Hz]')
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('RMY GIC PSD')
axes[1, 0].invert_xaxis()

# Plot 4: MZT PSD
axes[1, 1].plot(periods[valid_mask], psd_mzt_scipy[valid_mask])
axes[1, 1].set_xlabel('Period [hours]')
axes[1, 1].set_ylabel('PSD MZT GIC [A^2/Hz]')
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('MZT GIC PSD')
axes[1, 1].invert_xaxis()

# Adjust layout and display
plt.tight_layout()
plt.show()


#powerlaw
from scipy.optimize import curve_fit
# Power-law fitting function
def power_law_fit(x, a, b):
    return a * np.power(x, b)
# Fit the power-law curve
params, covariance = curve_fit(power_law_fit, psd_asy_scipy, yvals)
# Generate fitted values
fitted_yvals = power_law_fit(psd_asy_scipy, *params)