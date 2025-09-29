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

dt = 60.0  # sampling interval in seconds
fs = 1.0 / dt  # sampling frequency in Hz
fny = fs / 2.0  # Nyquist frequency

n = len(index['ASYH'])

asy = np.array(index['ASYH'])
lav = np.array(stat_dir['LAV']).flatten()
rmy = np.array(stat_dir['RMY']).flatten()
mzt = np.array(stat_dir['MZT']).flatten()
qro = np.array(stat_dir['QRO']).flatten()

lav_interp = interpolate_nan(lav)
rmy_interp = interpolate_nan(rmy)
mzt_interp = interpolate_nan(mzt)
qro_interp = interpolate_nan(qro)

# Or use scipy's built-in function (recommended)
freqs, psd_asy_scipy = signal.welch(asy, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_qro_scipy = signal.welch(qro_interp, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_lav_scipy = signal.welch(lav_interp, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_mzt_scipy = signal.welch(mzt_interp, fs=fs, window='hamming', nperseg=n, scaling='density')
freqs, psd_rmy_scipy = signal.welch(rmy_interp, fs=fs, window='hamming', nperseg=n, scaling='density')

periods = 1 / freqs / 3600  # Convert from seconds to hours

# Filter out infinite periods (from frequency = 0)
valid_mask = np.isfinite(periods) & (periods > 0)

# Manual absolute positioning in figure coordinates (0-1)
fig = plt.figure(figsize=(12, 10))

# Define positions: [left, bottom, width, height] in figure coordinates (0-1)
ax1_pos = [0.10, 0.70, 0.80, 0.25]  # ASYM: wide and tall
ax2_pos = [0.10, 0.40, 0.35, 0.25]  # LAV
ax3_pos = [0.55, 0.40, 0.35, 0.25]  # QRO  
ax4_pos = [0.10, 0.10, 0.35, 0.25]  # RMY
ax5_pos = [0.55, 0.10, 0.35, 0.25]  # MZT

# Create axes with exact positions
ax1 = fig.add_axes(ax1_pos)
ax2 = fig.add_axes(ax2_pos)
ax3 = fig.add_axes(ax3_pos)
ax4 = fig.add_axes(ax4_pos)
ax5 = fig.add_axes(ax5_pos)

# Now plot on each axis
ax1.plot(periods[valid_mask], psd_asy_scipy[valid_mask], color='red', linewidth=2)
ax1.set_xlabel('Period [hours]')
ax1.set_ylabel('PSD ASYM [nT²/Hz]')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('ASYM-H PSD - Geomagnetic Activity Index', fontsize=14, fontweight='bold')
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3)

ax2.plot(periods[valid_mask], psd_lav_scipy[valid_mask], color='blue')
#ax2.set_xlabel('Period [hours]')
ax2.set_ylabel('PSD LAV GIC [A²/Hz]')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('LAV GIC PSD')
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)

ax3.plot(periods[valid_mask], psd_qro_scipy[valid_mask], color='green')
#ax3.set_xlabel('Period [hours]')
ax3.set_ylabel('PSD QRO GIC [A²/Hz]')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_title('QRO GIC PSD')
ax3.invert_xaxis()
ax3.grid(True, alpha=0.3)

ax4.plot(periods[valid_mask], psd_rmy_scipy[valid_mask], color='purple')
ax4.set_xlabel('Period [hours]')
ax4.set_ylabel('PSD RMY GIC [A²/Hz]')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_title('RMY GIC PSD')
ax4.invert_xaxis()
ax4.grid(True, alpha=0.3)

ax5.plot(periods[valid_mask], psd_mzt_scipy[valid_mask], color='orange')
ax5.set_xlabel('Period [hours]')
ax5.set_ylabel('PSD MZT GIC [A²/Hz]')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_title('MZT GIC PSD')
ax5.invert_xaxis()
ax5.grid(True, alpha=0.3)

# No tight_layout needed since we manually positioned everything
plt.show()


sys.exit('end of process')
#powerlaw
from scipy.optimize import curve_fit
# Power-law fitting function
def power_law_fit(x, a, b):
    return a * np.power(x, b)
# Fit the power-law curve
params, covariance = curve_fit(power_law_fit, psd_asy_scipy, yvals)
# Generate fitted values
fitted_yvals = power_law_fit(psd_asy_scipy, *params)