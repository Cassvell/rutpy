import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data
from calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt

def corr_offset(data):
    ndata = len(data)
    
    hourly_data = np.array(ndata/60)

    sys.exit('end of child process')
    #for i in range(len(hourly_data)):

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
stat = ['LAV', 'QRO', 'RMY', 'MZT']

gic, T1, T2 = process_station_data(i_date, f_date, path, stat[0], idx1, tot_data)

plt.plot(gic.index, gic.values, label='GIC Data')
plt.show()