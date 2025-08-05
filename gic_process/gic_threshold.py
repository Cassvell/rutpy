
import matplotlib.pyplot as plt
from gicdproc import  process_station_data
from timeit import default_timer as timer

import pandas as pd
import os.path

import numpy as np
from datetime import datetime, timedelta
from calc_daysdiff import calculate_days_difference
from ts_acc import mz_score
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.dates as mdates
#from corr_offset import corr_offset
import sys
import os

def threshold(stddev_30):
    nbins = int(len(stddev_30) / 3)
    frequencies, bin_edges = np.histogram(stddev_30, bins=nbins*2, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gaussian, bin_centers, frequencies, 
                            p0=[1, np.mean(stddev_30), np.std(stddev_30)])

    x_fit = np.linspace(min(bin_edges), max(bin_edges), 500)    
    print('##################################################################')
    print('##################################################################')



        # Create figure with two subplots
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]}, sharex=False)

    # --- Top Plot: Time Series ---
    #ax1.plot(stddev_30.index, stddev_30.values, 'r-', linewidth=1.5, label='20-min Std Dev')

    # Format datetime x-axis
    #ax1.xaxis.set_major_locator(mdates.DayLocator())
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24,6)))
    #ax1.set_title(f"{stat}")
    #ax1.set_ylabel('Standard Deviation (mV/m)')
    #ax1.legend(loc='upper right')
    #ax1.grid(True, which='both', alpha=0.3)
    #fig.autofmt_xdate(rotation=45)  # Rotate date labels

    # --- Bottom Plot: CDF ---
    sorted_data = np.sort(stddev_30.values)
    cdf = np.arange(1, len(sorted_data)+1 / len(sorted_data))

    #ax2.plot(sorted_data, cdf, 'b-', linewidth=2, label='Empirical CDF')

    # Add Gaussian fit if available
    #if 'popt' in locals():
    #    x_fit = np.linspace(min(sorted_data), max(sorted_data), 100)
    #    ax2.plot(x_fit, norm.cdf(x_fit, popt[1], popt[2]), 'r--',
    #            label=f'Gaussian Fit (μ={popt[1]:.2f}, σ={popt[2]:.2f})')

    # Add threshold lines
    thresholds = {
        '2σ': (popt[1] + 2*popt[2], 'orange', '--'),
        '95%': (np.percentile(sorted_data, 95), 'purple', ':')
    }

    #for label, (value, color, ls) in thresholds.items():
    #    ax2.axvline(x=value, color=color, linestyle=ls, 
    #                label=f'{label} = {value:.2f} (mV/m)')
    #    ax2.plot(value, norm.cdf(value, popt[1], popt[2]), 'o', color=color)

    #ax2.set_xlabel('Standard Deviation Values (mV/m)')
    #ax2.set_ylabel('Cumulative Probability')
    #ax2.grid(True, alpha=0.3)
    #ax2.set_ylim(0, 1)

    sigma_threshold = popt[1] + 2*popt[2]  # Gaussian 2σ threshold
    percentile_95 = np.percentile(sorted_data, 95)  # Empirical 95th percentile

    # Compute simple average
    avg_threshold = (sigma_threshold + percentile_95) / 2
    cdf_value = norm.cdf(avg_threshold, popt[1], popt[2])
    #ax2.plot(avg_threshold, cdf_value, 'ko', markersize=8, label=f'threshold: {avg_threshold}')
    #ax2.legend(loc='lower right')
    #plt.tight_layout()
    #plt.savefig(f'/home/isaac/rutpy/gicsOutput/mothersday/threshold_{i_date}_{f_date}.png', dpi=300)
    #plt.show()
    indices = []
    
    for i in range(len(stddev_30)):
        if stddev_30[i] >= avg_threshold:
            indices.append(stddev_30.index[i])

    return avg_threshold, indices


'''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(gic[719:-721].index, gic[719:-721], label=f'preprocessed {stat[i].upper()}', alpha=0.8)
    ax1.plot(gic[719:-721].index, qd, label='Diurnal Variation', color='red')
    #ax1.scatter(exceed_indices, gic[exceed_indices], color='black', label='Threshold Exceedance')
    ax1.set_xlim(inicio, final)
    ax1.set_ylabel('GIC (A)')
    ax1.legend()

	# Plot difference
    ax2.plot(gic[719:-721].index, gic[719:-721]-qd, label=f'{stat[i].upper()} - Diurnal Variation', color='green')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('Difference (A)')
    ax2.set_xlim(inicio, final)
    ax2.legend()

    plt.tight_layout()
    #plt.savefig(f'/home/isaac/MEGA/posgrado/doctorado/semestre4/gics_procesados/{stat[i]}_{inicio.strftime('%Y-%m-%d')}_{final.strftime('%Y-%m-%d')}.png', dpi = 300)
    plt.show()

'''
