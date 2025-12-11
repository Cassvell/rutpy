import pandas as pd
import numpy as np 
from datetime import datetime
import re
import os
import sys


idate = sys.argv[1]
fdate = sys.argv[2]


idate_formatted = datetime.strptime(idate, '%Y%m%d').strftime('%Y-%m-%d')
fdate_formatted = datetime.strptime(fdate, '%Y%m%d').strftime('%Y-%m-%d')

file_path='/home/isaac/datos/PC/'
status = 'D'
file_name = f'{file_path}PCN_{idate_formatted}_{fdate_formatted}_{status}.dat'



if not os.path.isfile(file_name):
    status = 'P'
    file_name = f'{file_path}PCN_{idate_formatted}_{fdate_formatted}_{status}.dat'


df = pd.read_csv(file_name, header=25, sep=r'\s+')  # Ensure correct delimiter

ini = str(df.iloc[0, 0])
fin = str(df.iloc[-1, 0])
idx = pd.date_range(start=f"{ini} 00:00:00", end=f"{fin} 23:59:00", freq='min')



df['datetime'] = idx
df.set_index('datetime', inplace=True)

#drop_cols = [col for col in ['|', 'DATE', 'TIME', 'datetime'] if col in df.columns]
df = df.drop(columns=['|', 'DATE', 'TIME'])  

# Ensure all necessary columns exist
required_cols = ["DOY", "PCN"]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    exit()

# Generate daily files
idx_daily = pd.date_range(start=f"{ini} 00:00:00", end=f"{fin} 23:59:00", freq='D')
dates = idx_daily.strftime('%Y%m%d')
step = 1440  # 1440 minutes in a day

new_path = "/home/isaac/datos/PC/daily/"
os.makedirs(new_path, exist_ok=True)  # Ensure directory exists


for i in range(len(idx) // step):
    daily_data = df.iloc[i * step : (i + 1) * step]
    
    if daily_data.empty:
        print(f"Skipping day {i}, no data")
        continue

    # Ensure idx_daily is within range
    if i >= len(idx_daily):
        print(f"Skipping day {i}, idx_daily is out of range")
        continue  

    fname = f"pcn_{idx_daily[i].strftime('%Y%m%d')}.dat"
    full_path = os.path.join(new_path, fname)

    with open(full_path, 'w') as f:
        for _, row in daily_data.iterrows():
            try:
                # CORRECTED: Changed :f6.2 to :6.2f
                formatted_line = f"{float(row['PCN']):6.2f}\n"
                f.write(formatted_line)
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")

    print(f"Saved: {full_path}")
