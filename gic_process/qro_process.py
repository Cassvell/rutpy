import sys 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from gic_threshold import threshold
from gic_diurnalbase import gic_diurnalbase
from get_files import get_files, list_names, get_file

idate = sys.argv[1]
fdate = sys.argv[2]

iyear = int(idate[0:4])
imonth = int(idate[4:6])
iday = int(idate[6:8])


fyear = int(fdate[0:4])
fmonth = int(fdate[4:6])
fday = int(fdate[6:8])



idx1 = pd.date_range(start = pd.Timestamp(str(idate)+' 00:00:00' ), end =\
                        pd.Timestamp(fdate+' 23:59:00'), freq='min')

idx_daylist = pd.date_range(start = pd.Timestamp(str(idate)), \
                                        end = pd.Timestamp(str(fdate)), freq='D')
    
idx_list = (idx_daylist.strftime('%Y-%m-%d')) 
str1 = "GIC_"
ext = "_QRO.dat"

dir_path= '/home/isaac/datos/gics_obs/jur2024'
list_fnames = list_names(idx_list, str1, ext)

dfs_c = []
missing_vals = ["NaN", "NO DATA"] 

for i in range(len(list_fnames)):      
    year = idx_daylist[i].year  # Extract year directly from the Timestamp
    SG2 = f"{dir_path}/"
    
        # Read first 5 rows
    df_c = pd.read_csv(SG2+list_fnames[i], header=0, sep='\t',parse_dates = [0], na_values = missing_vals)
    
    # Count empty values
    empty_counts = df_c.isna().sum()  # Count NaN values
    empty_strings = (df_c == '').sum()  # Count empty strings
    total_empty = empty_counts + empty_strings
    
    # Check if any empty values exist (sum across all columns)
    if total_empty.sum() > 0:
        print(f"Found {total_empty.sum()} empty values to replace")
        
        # Replace both NaN and empty strings
        df_c.replace({'': -999.999}, inplace=True)
        df_c.fillna(-999.999, inplace=True)
    dfs_c.append(df_c)  
    
df = pd.concat(dfs_c, axis=0, ignore_index=True)
    
df = df.replace(-999.999, np.nan)   
df = df.set_index(idx1)     
#   idx2 = pd.date_range(start = pd.Timestamp(date1), \
#                                    end = pd.Timestamp(date2), freq='H')  




gic_res, qd = gic_diurnalbase(df['gic_corr'], idate, fdate, 'qro')   

#plt.plot(df['gic_corr'])
plt.plot(df.index, qd) 
plt.plot(df.index, gic_res, color='green')
plt.show()

output_path = f'/home/isaac/datos/gics_obs/processed/{fyear}/'

ndays = int(len((df['gic_corr']))/1440)
for j in range(ndays):
    start_idx = j * 1440
    end_idx = (j + 1) * 1440
    daily_data = df['gic_corr'].iloc[start_idx:end_idx]
    daily_data.fillna(9999.9999, inplace=True)
    date_str = daily_data.index[0].strftime('%Y%m%d')
    daily_data.to_csv(output_path + f'QRO/gic_QRO_{date_str}.csv', header=False)