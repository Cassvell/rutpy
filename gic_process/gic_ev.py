import matplotlib.pyplot as plt
from gicdproc import pproc, reproc, df_gic, df_dH, df_Kloc, fix_offset, process_station_data
from timeit import default_timer as timer
import sys
import pandas as pd
import os.path
import os
import numpy as np
from datetime import datetime, timedelta
from calc_daysdiff import calculate_days_difference
from ts_acc import mz_score
from gic_threshold import threshold
import matplotlib.dates as mdates
from datetime import datetime, timedelta


def round_to_10_minutes(dt):
    if dt is None:
        return None
    
    # Calculate minutes to add/subtract for rounding
    remainder = dt.minute % 10
    if remainder < 5:
        # Round down (e.g., 24→20)
        rounded_minute = dt.minute - remainder
    else:
        # Round up (e.g., 27→30)
        rounded_minute = dt.minute + (10 - remainder)
    
    # Handle hour overflow if rounding up past 59 minutes
    if rounded_minute >= 60:
        return (dt.replace(hour=dt.hour+1, minute=0, second=0, microsecond=0)
                .replace(minute=0))
    else:
        return dt.replace(minute=rounded_minute, second=0, microsecond=0)

# Get the absolute path to the directory containing your module
module_dir = os.path.abspath('/home/isaac/rutpy') 
sys.path.append(module_dir)

# Now you can import the module
from pick_range import on_move, on_click

        
idate = sys.argv[1]
fdate = sys.argv[2]

stat = ['QRO', 'LAV', 'RMY', 'MZT']
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
        #file2 = os.path.join(data_dir, s, f'gic_{s}_{day}_reproc.csv')
        if os.path.isfile(file):
            df = pd.read_csv(file, index_col=0, header=None, parse_dates=True)
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


clicked_dates = []

def on_click(event):
    if event.xdata is not None:
        clicked_datetime = mdates.num2date(event.xdata)
        clicked_dates.append(clicked_datetime)
        print(f"Clicked at: {clicked_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Disconnect after 2 clicks
        if len(clicked_dates) >= 2:
            plt.disconnect(cid)
            plt.close()


# Create figure and plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))

# Plot data for each station
stations = {
    'LAV': ('blue', ax1),
    'QRO': ('orange', ax2),
    'RMY': ('green', ax3),
    'MZT': ('purple', ax4)
}

for name, (color, ax) in stations.items():
    ax.plot(stat_dir[name].index, stat_dir[name], label=name, color=color, alpha=0.7)
    ax.set_xlim(stat_dir[name].index[0], stat_dir[name].index[-1])
    ax.legend()
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# Connect click event
cid = fig.canvas.mpl_connect('button_press_event', on_click)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

if len(clicked_dates) >= 2:
    start_date, end_date = sorted(clicked_dates[:2])
    
    # Round to nearest 10 minutes
    rounded_start = round_to_10_minutes(start_date)
    rounded_end = round_to_10_minutes(end_date)
    
    # Format without timezone
    fmt_start = rounded_start.strftime('%Y-%m-%d %H:%M:%S')
    fmt_end = rounded_end.strftime('%Y-%m-%d %H:%M:%S')
    
    print("\nSelected time range:")
    print(f"Original - Start: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Original - End: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rounded - Start: {fmt_start}")
    print(f"Rounded - End: {fmt_end}")
    
    # Extract data within zoom range
    zoom_data = {}
    for station in stat_dir.keys():
        tmp = np.isnan(stat_dir[station].values).all()
        if not tmp:
            mask = (stat_dir[station].index >= fmt_start) & \
                    (stat_dir[station].index <= fmt_end)
            zoom_data[station] = stat_dir[station][mask]
            print(f"Data for {station}")
               
        else:
            print(f"All values are NaN for {station}")
else:
    print("\n\n CHINGA TU MADRE, NO SELECCIONASTE NADA!!!\n\n")

threshold_lav, indices = threshold(zoom_data['LAV'].resample('10 min').median())
print(threshold_lav)

sys.exit('End of script')





threshold_dir = {'LAV': threshold(stat_dir['QRO'].resample('15 min').median()), 
                 'QRO': threshold(stat_dir['QRO'].resample('15 min').median()),
                 'RMY': threshold(stat_dir['RMY'].resample('15 min').median()),
                 'MZT': threshold(stat_dir['MZT'].resample('15 min').median())}
    