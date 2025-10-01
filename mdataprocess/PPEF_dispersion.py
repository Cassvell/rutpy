import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from threshold import max_IQR
from scipy.stats import genpareto, kstest #anderson
import kneed as kn
from lmoments3 import distr
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import halfnorm, rayleigh
#st= sys.argv[1]
idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]
tot = sys.argv[3]
sector = ['TW1', 'TW2']

path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
#st = ['lzh', 'bmt', 'tam', 'jai', 'cyg', 'teo', 'hon', 'gui',  'kak', 'sjg']
#color = ['red', 'olive', 'seagreen', 'magenta', 'purple', 'orange', 'darkgreen', 'salmon', 'sienna', 'gray']
st_sect = ['gui', 'jai', 'kak', 'teo']
color = ['red', 'orange', 'seagreen', 'purple']
path = '/home/isaac/datos/pca/'
#period = ['1101-2300 LT', '1801-0600 LT', '2201-1000 LT', '0601-1800 LT']

for i in range(len(st_sect)):
    print(f'Observatorio: {st_sect[i]}')
    df = pd.read_csv(f'{path}{st_sect[i]}_{idate}_{fdate}_TWtot.dat', sep='\\s+')
    dp2 = df.iloc[:, 2]
    
    missing_count = 2880 - len(dp2)
    zeros_series = pd.Series(np.zeros(missing_count))
    dp2 = pd.concat([dp2, zeros_series], ignore_index=True)
    
    dp2_picks = []
    resample = 15
    num_windows = 2880 // resample
    
    for j in range(num_windows):
        start_idx = j * resample
        end_idx = (j + 1) * resample
        tmp = np.nanmax(np.abs(dp2[start_idx:end_idx]))        
        dp2_picks.append(tmp)
    
    dp2_app = []
    for k in sector:
    # Read data
        df2 = pd.read_csv(f'{path}{st_sect[i]}_{idate}_{fdate}_{k}.dat', header=None,sep='\\s+')
        dp2_tmp = df2.iloc[:, 2]
        
        dp2_app.append(dp2_tmp)

    dp2_2 = pd.concat(dp2_app, ignore_index=True) 
    dp2_2 = dp2_2[:-1]

    dp2_picks_2 = []
    num_windows2 = len(dp2_2) // resample
    
    for l in range(num_windows2):
        start_idx = l * resample
        end_idx = (l + 1) * resample
        tmp = np.nanmax(np.abs(dp2[start_idx:end_idx]))        
        dp2_picks_2.append(tmp)   

    nbins = int(len(dp2))
    dp2 = np.array(dp2)  
    dp2_2 = np.array(dp2_2)  
    
    # Clean data
    dp2 = dp2[~np.isnan(dp2)]
    dp2 = np.unique(dp2)
    
    dp2_2 = dp2_2[~np.isnan(dp2_2)]
    dp2_2 = np.unique(dp2_2)    
    
    # Histogram and Gaussian fit
    frequencies, bin_edges = np.histogram(dp2, bins=nbins*4, density=True)
    
    frequencies_2, bin_edges_2 = np.histogram(dp2_picks, bins=int(len(dp2_picks) / 3), density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    
    bin_centers_2 = (bin_edges_2[:-1] + bin_edges_2[1:]) / 2 
    
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, bin_centers, frequencies, 
                        p0=[1, np.mean(dp2), np.std(dp2)])
  
    x_fit = np.linspace(min(bin_edges), max(bin_edges), 500)
    x_fit2 = np.linspace(min(bin_edges_2), max(bin_edges_2), 500)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    # --- Top plot: PDF of ORIGINAL data ---
    ax1.hist(dp2, density=True, bins=int(len(dp2) / 15), color='navy', 
            histtype='stepfilled', alpha=0.4, label=r'$\mathrm{H_{PPEF}}$ distribution')
    ax1.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, 
            label=f'Gaussian fit: μ={popt[1]:.2f}, σ={popt[2]:.2f}')
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(0, 0.1)
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    station_name = st_sect[i].upper()
    ax1.set_title(f"{station_name}: {idate} to {fdate}")

    # --- Bottom plot: CDF of RESAMPLED data (dp2_picks) ---
    sorted_data = np.sort(np.abs(dp2))
    sorted_data_2 = np.sort(np.abs(dp2_2))

    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    cdf2 = np.arange(1, len(sorted_data_2)+1) / len(sorted_data_2)

    # Plot empirical CDF of resampled data
    ax2.plot(sorted_data, cdf, 'b-', label=r'$max[|H_{PPEF}|]$ CDF', linewidth=3)

    # Plot Gaussian CDF fit (using parameters from ORIGINAL data fit)
    # Note: This uses the Gaussian parameters from the original data to fit the resampled CDF

    params = halfnorm.fit(sorted_data)
    mean, std = params
    ax2.plot(x_fit, halfnorm.cdf(x_fit, *params), 'r-', linewidth=2)
    percentile = 0.95
    idx = np.searchsorted(cdf, percentile)
    idx_2 = np.searchsorted(cdf2, percentile)

    if idx < len(sorted_data):
        # Add vertical line at the 95% value for resampled data
        value_95 = sorted_data[idx]
        ax2.axvline(x=value_95, color='red', linestyle='--', alpha=1, linewidth=1.5, 
                label=f'Threshold = {value_95:.2f} nT')
    
    if idx_2 < len(sorted_data_2):    
        value_95_2 = sorted_data_2[idx_2]
        ax2.axvline(x=value_95_2, color='darkred', linestyle='--', alpha=1, linewidth=1.5, 
                label=f'Threshold (TW) = {value_95_2:.2f} nT')
        
        print(f'Threshold resampled: {value_95}, Threshold in TW: {value_95_2}')        

    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$\mathrm{H_{PPEF}}$ distribution [nT]")
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if tot == 'True': 
        plt.savefig(f'{path2}{st_sect[i]}_{idate}_{fdate}.V1.png', dpi=300)
        plt.show()
    else:
        plt.savefig(f'{path2}{st_sect[i]}_{idate}_{fdate}.V2.png', dpi=300)
        plt.show()