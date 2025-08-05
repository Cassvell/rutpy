import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data
from calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt
from gic_threshold import threshold
from gic_diurnalbase import gic_diurnalbase
from corr_offset import corr_offset
idate = sys.argv[1]
fdate = sys.argv[2]

fyear = int(fdate[0:4])
fmonth = int(fdate[4:6])
fday = int(fdate[6:8])

stat = ['LAV', 'QRO', 'RMY', 'MZT']

path = f'/home/isaac/datos/gics_obs/'

finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]
idx1 = pd.date_range(start = pd.Timestamp(idate+ ' 12:01:00'), \
                          end = pd.Timestamp(nextday + ' 12:00:00'), freq='min')

ndays = calculate_days_difference(idate, fdate)
tot_data = (ndays+1)*1440

file = []

gic_lav, T1TW_lav, T2TW_lav = process_station_data(idate, fdate, path, stat[0], idx1, tot_data)

#gicTW_qro, T1TW_qro, T2TW_qro = process_station_data(i_date, f_date, path2, stat[0], idx1, tot_data)

#gicTW_mzt, T1TW_mzt, T2TW_mzt = process_station_data(i_date, f_date, path2, stat[3], idx1, tot_data)

#gicTW_rmy, T1TW_rmy, T2TW_rmy = process_station_data(i_date, f_date, path2, stat[2], idx1, tot_data)

gic_lav_res, qd  = gic_diurnalbase(gic_lav, stat[0])

gic_lav_h = gic_lav_res.resample('20Min').median().fillna(method='ffill')

#sys.exit('end of child process')
threshold = threshold(gic_lav_h)   
#print(f'threshold: {threshold[0]}')
gic_lav_corroffset = corr_offset(gic_lav_res, threshold[0])


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows, 1 column

# First subplot
ax1.plot(gic_lav.index, gic_lav, color='blue', label='Despiked GIC LAV')
ax1.plot(gic_lav[719:-721].index, qd, color='red', label='diurnal variation')
ax1.legend()
ax1.set_title('Despiked GIC LAV')

# Second subplot
ax2.plot(gic_lav_res, color='green', label='GIC LAV - Diurnal Variation')
ax2.legend()
ax2.set_title('Diurnal Variation Removed')

# Third subplot
ax3.plot(gic_lav_corroffset, color='black', label='LAV Offset Jumps Corrected')
ax3.legend()
ax3.set_title('Offset Jumps Corrected')

plt.tight_layout()  # Prevents label overlapping
plt.show()