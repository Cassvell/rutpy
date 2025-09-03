import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]

file_path='/home/isaac/datos/ae/'
df = pd.read_csv(file_path+'AEAUALAO_'+idate+'_'+fdate+'m_P'+'.dat', header=24, sep=r'\s+')  # Ensure correct delimiter

# Drop unwanted columns (check if they exist)

# Create datetime index
ini = str(df.iloc[0, 0])
fin = str(df.iloc[-1, 0])
idx = pd.date_range(start=f"{ini} 00:00:00", end=f"{fin} 23:59:00", freq='T')



df['datetime'] = idx
df.set_index('datetime', inplace=True)

#drop_cols = [col for col in ['|', 'DATE', 'TIME', 'datetime'] if col in df.columns]
df = df.drop(columns=['|', 'DATE', 'TIME'])  


# Ensure all necessary columns exist
required_cols = ["AE", "AU", "AL", "AO"]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    exit()

# Generate daily files
idx_daily = pd.date_range(start=f"{ini} 00:00:00", end=f"{fin} 23:59:00", freq='D')
dates = idx_daily.strftime('%Y%m%d')
step = 1440  # 1440 minutes in a day

new_path = "/home/isaac/datos/ae/daily/"
os.makedirs(new_path, exist_ok=True)  # Ensure directory exists


for i in range(len(idx) // step):
    daily_data = df.iloc[i * step : (i + 1) * step]
    
    if daily_data.empty:
        print(f"Skipping day {i}, no data")
        continue  # Skip empty days

    # Ensure idx_daily is within range
    if i >= len(idx_daily):
        print(f"Skipping day {i}, idx_daily is out of range")
        continue  

    fname = f"ae_{idx_daily[i].strftime('%Y%m%d')}.dat"
    full_path = os.path.join(new_path, fname)

    with open(full_path, 'w') as f:
        for _, row in daily_data.iterrows():
            try:
                formatted_line = f"{int(row['AE']):6d} {int(row['AU']):6d} {int(row['AL']):6d} {int(row['AO']):6d}\n"
                f.write(formatted_line)
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")

    print(f"Saved: {full_path}")

