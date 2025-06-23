import pandas as pd
import sys
import os
from magdata_processing import get_diurnalvar, base_line
from plots import plot_process
from aux_time_DF import index_gen, convert_date
from magnetic_datstruct import get_dataframe

net = sys.argv[1]
st= sys.argv[2]
idate = sys.argv[3]# "formato(yyyymmdd)"
fdate = sys.argv[4]

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
if net == 'regmex':
    #path = f"/home/isaac/datos/{net}/{st}/{st}20240/" # magnetic data path
    path = f"/home/isaac/datos/{net}/{st}20240/" # magnetic data path
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        new_name = str(date_name_newf)[0:8]
        
        fname = f"{st}{new_name}rK.min"
        #fname = f"{st}_{new_name}.clean"
        fname2 = st+'_'+new_name+'M.dat'
        fname3 = st+'_'+new_name+'M_XYZ.dat'
        filenames.append(fname)
        filenames_out.append(fname2)
        filenames_out2.append(fname3)
else:
    year = '2015'
    st_dir = st.upper()
    path = f"/home/isaac/datos/{net}/{year}/{st_dir}/" # magnetic data path
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        new_name = str(date_name_newf)[0:8]
        fname = st+date_name_newf+'dmin.min'
        #print(fname)
        fname2 = st+'_'+new_name+'.dat'
        fname3 = st+'_'+new_name+'_XYZ.dat'
        filenames.append(fname)
        filenames_out.append(fname2)
        filenames_out2.append(fname3)


magdata = get_dataframe(filenames, path, idx, dates, net)

H = magdata['H']
X = magdata['X']
Y = magdata['Y']
Z = magdata['Z']
D = magdata['D']
I = magdata['I']

baseline_curve = base_line(H, net, st)
base_lineX = base_line(X, net, st)
base_lineY = base_line(Y, net, st)
base_lineZ = base_line(Z, net, st)

H_detrend = H-baseline_curve
X_detrend = X-base_lineX
Y_detrend = Y-base_lineY
Z_detrend = Z-base_lineZ
#diurnal base line

diurnal_baseline, offset = get_diurnalvar(H_detrend, idx_daily, net, st)
diurnal_baselineX, offsetX = get_diurnalvar(X_detrend, idx_daily, net, st)
diurnal_baselineY, offsetY = get_diurnalvar(Y_detrend, idx_daily, net, st)
diurnal_baselineZ, offsetZ = get_diurnalvar(Z_detrend, idx_daily, net, st)

H_raw = H

H = H_detrend-diurnal_baseline
X = X_detrend - diurnal_baselineX
Y = Y_detrend - diurnal_baselineY
Z = Z_detrend - diurnal_baselineZ

H_noff1 = H-offset
X_noff1 = X-offsetX
Y_noff1 = Y-offsetY
Z_noff1 = Z-offsetZ

#sys.exit('end of the child process')
dst = []
hr = int(len(H)/60)
for i in range(hr):
    tmp_h = np.nanmedian(H_noff1[i*60:(i+1)*60])
    dst.append(tmp_h)
    
plot_process(H, H_raw, H_detrend, H_noff1, dst, baseline_curve, diurnal_baseline, st, idx_hr)

# Data dictionaries
dat = {'H': H_noff1, 'baseline_line': baseline_curve, 'SQ': diurnal_baseline}
dat2 = {'X': X_noff1, 'Y': Y_noff1, 'Z': Z_noff1, 'D': D, 'I': I}

# Create DataFrames
df = pd.DataFrame(dat).fillna(9999.9)   # Ensure NaN replacement
df2 = pd.DataFrame(dat2).fillna(9999.9) # Ensure NaN replacement




# Define path
path = f"/home/isaac/datos/{net}/{st}/minV2/"  
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
    daily_data2 = df2[start_idx:end_idx].reset_index(drop=True)

    # Define filenames
    full_path = os.path.join(path, filenames_out[i])
    full_path2 = os.path.join(path, filenames_out2[i])

    filenames.append(full_path)

    # Write first file
    with open(full_path, 'w') as f:
        for _, row in daily_data.iterrows():
            line = f"{row['H']:7.2f}{row['baseline_line']:10.2f}{row['SQ']:6.2f}\n"
            f.write(line)

    print(f"Saved: {full_path}")

    # Write second file
    with open(full_path2, 'w') as f2:
        for _, row in daily_data2.iterrows():
            line = f"{row['X']:7.2f}{row['Y']:8.2f}{row['Z']:8.2f}{row['D']:6.2f}{row['I']:6.2f}\n"
            f2.write(line)  # FIXED: Now correctly writes to f2

    print(f"Saved: {full_path2}")   
#df.to_csv()
