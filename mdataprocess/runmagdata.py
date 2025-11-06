import pandas as pd
import numpy as np
import sys
import os
from magdata_processing import get_diurnalvar, base_line
from plots import plot_process
from aux_time_DF import index_gen, convert_date
from magnetic_datstruct import get_dataframe
import matplotlib.pyplot as plt
net = sys.argv[1]
st= sys.argv[2]
data_class = sys.argv[3]
idate = sys.argv[4]# "formato(yyyymmdd)"
fdate = sys.argv[5]


year = int(idate[0:4])

enddata = fdate+ ' 23:59:00'
idx = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='min')
idx_hr = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='h')    
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')
filenames = []
filenames_out = []
filenames_out2 = []
dates = []
path = ''


def new_namedate(date):
    date = str(date)
    year = date[2:4]
    month = date[4:6]
    day = date[6:8]
    
    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_idx = int(month)-1
   # print(f'{year}, {month_names[month_idx]}, {day}')
    return year, month_names[month_idx], day


if net == 'regmex':
    #
    
    if data_class == 'preprocessed':
        path = f"/home/isaac/datos/{net}/{st}/{st}_raw/" # magnetic data path
    elif data_class == 'raw':
        path = f"/home/isaac/datos/{net}/{st}/raw/" # magnetic data path
    
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        year, month, day = new_namedate(date_name_newf)
        new_name = str(date_name_newf)[0:8]
        
        if data_class == 'preprocessed':
            fname = f"{st}_{new_name}.clean.dat"
        elif data_class == 'raw':
            if st == 'teo' or st == 'jur':
                fname = f"{st}{new_name}rK.min"
               
            elif st == 'coe' or st == 'itu':
                fname = f"{st}{day}{month}.{year}m"
       # print(path+fname)
        fname2 = st+'_'+new_name+'M.dat'
        fname3 = st+'_'+new_name+'M_XYZ.dat'
        filenames.append(fname)
        filenames_out.append(fname2)
        filenames_out2.append(fname3)
else:
    st_dir = st.upper()
    path = f"/home/isaac/datos/{net}/{year}/{st_dir}/" # magnetic data path
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        new_name = str(date_name_newf)[0:8]
        fname = st+date_name_newf+'dmin.min'
        #print(fname)
        fname2 = st+'_'+new_name+'m.dat'
        fname3 = st+'_'+new_name+'_XYZ.dat'
        filenames.append(fname)
        filenames_out.append(fname2)
        filenames_out2.append(fname3)



magdata = get_dataframe(filenames, st, data_class,path, idx, dates, net)
#magdata = get_dataframe(filenames, path, idx, dates, net)
H = magdata['H']
X = magdata['X']
Y = magdata['Y']
Z = magdata['Z']
D = magdata['D']
I = magdata['I']


#base_lineX = base_line(X, net, st)
#base_lineY = base_line(Y, net, st)
baseline_curve = base_line(H, net, st, '2s')
base_lineZ = base_line(Z, net, st, '2s')
base_lineD = base_line(D, net, st, '2s')


D_detrend = D-base_lineD
H_detrend = H-baseline_curve
#X_detrend = X-base_lineX
#Y_detrend = Y-base_lineY
Z_detrend = Z-base_lineZ
D_detrend = D-base_lineD

#diurnal base line

diurnal_baseline, offset = get_diurnalvar(H_detrend, idx_daily, net, st, qd_method='qd5', threshold_method='2s')
diurnal_baselineZ, offsetZ = get_diurnalvar(Z_detrend, idx_daily, net, st, qd_method='qd5', threshold_method='2s')
diurnal_baselineD, offsetD = get_diurnalvar(D_detrend, idx_daily, net, st, qd_method='qd5', threshold_method='2s')
#diurnal_baselineX, offsetX = get_diurnalvar(X_detrend, idx_daily, net, st)
#diurnal_baselineY, offsetY = get_diurnalvar(Y_detrend, idx_daily, net, st)

H_raw = H
Z_raw = Z
D_raw = D


D = D_detrend - diurnal_baselineD
Z = Z_detrend - diurnal_baselineZ
H = H_detrend-diurnal_baseline
#X = X_detrend - diurnal_baselineX
#Y = Y_detrend - diurnal_baselineY



#sys.exit('end of child process')

H_noff1 = H-offset
#X_noff1 = X-offsetX
#Y_noff1 = Y-offsetY
Z_noff1 = Z-offsetZ
D_noff1 = D-offsetD

#sys.exit('end of the child process')
dst = []
hr = int(len(H)/60)
for i in range(hr):
    tmp_h = np.nanmedian(H_noff1[i*60:(i+1)*60])
    dst.append(tmp_h)


def hourly(data):
    hourly_data = []
    hr = int(len(data)/60)
    for i in range(hr):
        tmp = np.nanmedian(data[i*60:(i+1)*60])
        hourly_data.append(tmp)
    
    return hourly_data

H_hr = hourly(H_noff1)
Z_hr = hourly(Z_noff1)
D_hr = hourly(D_noff1)

#plt.plot(idx, Z_raw, color='k')
#plt.plot(idx, base_lineZ, color = 'r')
#plt.show()


#plt.plot(idx, D_raw, color='k')
#plt.plot(idx, base_lineD, color = 'r')
#plt.show()


#sys.exit('end of test')
    
plot_process(H, H_raw, H_detrend, H_noff1, H_hr, baseline_curve, diurnal_baseline, st, idx_hr, 'H')
plot_process(D, D_raw, D_detrend, D_noff1, D_hr, base_lineD, diurnal_baselineD, st, idx_hr, 'D')
plot_process(Z, Z_raw, Z_detrend, Z_noff1, Z_hr, base_lineZ, diurnal_baselineZ, st, idx_hr, 'Z')

H = H_noff1+baseline_curve+diurnal_baseline
D = D_noff1+base_lineD+diurnal_baselineD
Z = Z_noff1+base_lineZ+diurnal_baselineZ
# Data dictionaries
dat = {'H': H_noff1,
       'D': D_noff1, 
       'Z': Z_noff1,
       'H base line': baseline_curve, 
       'D base line': base_lineD, 
       'Z base line': base_lineZ, 
       'H SQ': diurnal_baseline,
       'D SQ': diurnal_baselineD,
       'Z SQ': diurnal_baselineZ,}


#dat2 = {'X': X_noff1, 'Y': Y_noff1, 'Z': Z_noff1, 'D': D, 'I': I}

# Create DataFrames
df = pd.DataFrame(dat).fillna(999.9)   # Ensure NaN replacement
#df2 = pd.DataFrame(dat2).fillna(9999.9) # Ensure NaN replacement



header = " ".join(f"{key:>10}" for key in dat.keys())

header = f"{'H':>7}{'D':>14}{'Z':>13}{'Hbase':>12}{'Dbase':>10}{'Zbase':>12}{'HSQ':>8}{'DSQ':>10}{'ZSQ':>12}"

# Define path
path = f"/home/isaac/datos/{net}/{st}/experiment_{st}/"  
os.makedirs(path, exist_ok=True)  # Creates directory if it does not exist

# Iterate over daily indexes
for i in range(len(idx_daily)):
    start_idx = i * 1440
    end_idx = (i + 1) * 1440

    # Check if indices are within bounds
    if end_idx > len(df):
        print(f"Skipping {i}, index out of range")
        continue

    # Slice daily data
    daily_data = df[start_idx:end_idx].reset_index(drop=True)

    # Define filename
    full_path = os.path.join(path, filenames_out[i])
    filenames.append(full_path)

    # Write file with header
    with open(full_path, 'w') as f:
        # Write header
        f.write(header + '\n')
        
        # Write data rows
        for _, row in daily_data.iterrows():
            line = f"{row['H']:10.2f}{row['D']:16.10f}{row['Z']:10.2f}{row['H base line']:10.2f}{row['D base line']:12.8f}{row['Z base line']:12.2f}{row['H SQ']:6.2f}{row['D SQ']:14.8f}{row['Z SQ']:8.2f}\n"
            f.write(line)

    print(f"Saved: {full_path}")