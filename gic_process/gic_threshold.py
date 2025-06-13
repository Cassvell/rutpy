
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

import sys
import os

# Get the absolute path to the directory containing your module
module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

# Now you can import the module
#import magdata_processing 


from magdata_processing import get_diurnalvar, get_qd_dd

stat = ['LAV', 'QRO', 'RMY', 'MZT']
def threshold(stddev_30, stat):
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


i_date = sys.argv[1] 

f_date = sys.argv[2]


fyear = int(f_date[0:4])
fmonth = int(f_date[4:6])
fday = int(f_date[6:8])


finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]

idx1 = pd.date_range(start = pd.Timestamp(i_date+ ' 12:01:00'), \
                          end = pd.Timestamp(nextday + ' 12:00:00'), freq='min')


ndays = calculate_days_difference(i_date, f_date)
tot_data = (ndays+1)*1440


path = '/home/isaac/datos/gics_obs/'
file = []

#gicTW_lav, T1TW_lav, T2TW_lav = process_station_data(i_date, f_date, path, stat[1], idx1, tot_data)

#gicTW_qro, T1TW_qro, T2TW_qro = process_station_data(i_date, f_date, path, stat[0], idx1, tot_data)

#gicTW_mzt, T1TW_mzt, T2TW_mzt = process_station_data(i_date, f_date, path, stat[3], idx1, tot_data)

#gicTW_rmy, T1TW_rmy, T2TW_rmy = process_station_data(i_date, f_date, path, stat[2], idx1, tot_data)


for i in range(len(stat)):
    gic, T1, T2 = process_station_data(i_date, f_date, path, stat[i], idx1, tot_data)
	
    stddev_20 = gic.resample('20Min').std().fillna(method='ffill')
    threshold_value, indices = threshold(stddev_20, stat[i])
    exceed_indices = stddev_20[stddev_20 > threshold_value].index
    
    idx_daily = pd.date_range(start = gic.index[719], end = gic.index[-722], freq= 'D' )
    qd, offset = get_diurnalvar(gic[719:-721], idx_daily, stat[i].lower())
    
    inicio = gic.index[719]
    final = gic.index[-721]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(gic[719:-721].index, gic[719:-721], label=f'preprocessed {stat[i].upper()}', alpha=0.8)
    ax1.plot(gic[719:-721].index, qd, label='Diurnal Variation', color='red')
    ax1.scatter(exceed_indices, gic[exceed_indices], 
           color='black', label='Threshold Exceedance')
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
    plt.savefig(f'/home/isaac/MEGA/posgrado/doctorado/semestre4/gics_procesados/{stat[i]}_{inicio.strftime('%Y-%m-%d')}_{final.strftime('%Y-%m-%d')}.png', dpi = 300)
    plt.show()


#inicio = gicTW_lav.index[0]
#final  = gicTW_lav.index[-1]

print('##################################################################')
print('##################################################################')

#stddev_30 = gicTW_lav.resample('20Min').std().fillna(method='ffill')

#threshold_value, indices = threshold(stddev_30, stat[1])  # Assuming threshold() is defined
#exceed_indices_lav = stddev_30[stddev_30 > threshold_value].index

#normal_periods = ~gicTW_lav.index.isin(exceed_indices_lav)
'''
#fig, ax = plt.subplots(4, figsize=(12,14))
#fig.suptitle('Estudio de GICs', fontsize=24, fontweight='bold')

#ax[0].plot(gicTW_lav)
#ax[0].grid()
#ax[0].set_xlim(inicio,final)
#ax[0].set_title('LAV st', fontsize=18)
#ax[0].set_ylabel(' GIC [A]', fontweight='bold')

#ax[1].plot(gicTW_qro)
#ax[1].grid()
#ax[1].set_xlim(inicio,final)
#ax[1].set_title('QRO st', fontsize=18)
#ax[1].set_ylabel(' GIC [A]', fontweight='bold')

#ax[2].plot(gicTW_rmy)
#ax[2].grid()
#ax[2].set_xlim(inicio,final)
#ax[2].set_title('RMY st', fontsize=18)
#ax[2].set_ylabel(' GIC [A]', fontweight='bold')

#ax[3].plot(gicTW_mzt)
#ax[3].grid()
#ax[3].set_xlim(inicio,final)
#ax[3].set_title('MZT st', fontsize=18)
#ax[3].set_ylabel(' GIC [A]', fontweight='bold')

#fig.tight_layout()
#plt.savefig(f'/home/isaac/rutpy/gicsOutput/mothersday/gic_stat_{i_date}_{f_date}.png', dpi=300)
#plt.show()
'''    


