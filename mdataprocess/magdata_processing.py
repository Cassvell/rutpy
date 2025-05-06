import pandas as pd
import numpy as np
#from statistics import mode
#from datetime import datetime
# Ajuste de distribuciones
import sys
#from numpy.linalg import LinAlgError
#from scipy.interpolate import splrep, splev
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter1d
#from scipy.interpolate import NearestNDInterpolator
from magnetic_datstruct import get_dataframe
from scipy.signal import medfilt
from aux_time_DF import index_gen, convert_date
from lowpass_filter import aphase, dcomb
from typical_vall import night_hours, mode_nighttime, typical_value, gaus_center, mode_hourly
from threshold import get_threshold, max_IQR, med_IQR
from plots import plot_GPD, plot_detrend, plot_qdl ,plot_process
#from Ffitting import fit_data
from night_time import night_time
import os
from scipy import fftpack, signal
###############################################################################
#generación del índice temporal Date time para las series de tiempo
###############################################################################   
###############################################################################
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
    path = f"/home/isaac/datos/{net}/{st}/{st}_raw/" # magnetic data path
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        new_name = str(date_name_newf)[0:8]
        fname = st+'_'+new_name+'.clean.dat'
        fname2 = st+'_'+new_name+'.dat'
        fname3 = st+'_'+new_name+'_XYZ.dat'
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
     
###############################################################################}
###############################################################################
#Base line Determination
###############################################################################
###############################################################################
###############################################################################

###############################################################################
#Monthly base line
###############################################################################   
def base_line(data, idx, idx_daily):    
    ndata = len(data)
###############################################################################
#Typical Day computation
###############################################################################  
    daily_mode = mode_nighttime(data, 60, net, st)
    night_data = night_hours(data, net, st)   
    

    daily_gauss = []

    for i in range(len(daily_mode)):
        tmp = night_data[i*180: ((i+1)*180)-1]
        tmp_gauss = gaus_center(tmp)
        daily_gauss.append(tmp_gauss)

    flat_daily_gauss = [item for sublist in daily_gauss for item in sublist]

    daily_sample = len(data)/1440

    daily_stacked = typical_value(daily_mode, flat_daily_gauss, daily_sample)
###############################################################################
###############################################################################
#Use of threshold for identify and Isolate disturbed days from non disturbed
###############################################################################
###############################################################################
    #We determine first an array of variation picks using Inter Quartil Range
    pickwindow = [3,4]
    original_daily_stacked = np.copy(daily_stacked)

    picks = max_IQR(data, 60, pickwindow[0], method='iqr')
    
    x, GPD, knee, threshold = get_threshold(picks)

    # Validate GPD fit using the second derivative
    second_derivative = np.gradient(np.gradient(GPD))

    # Daily IQR picks and classification
    daily_picks = med_IQR(data, 60, 24, method='iqr')

    for j in range(len(daily_stacked)):
        # Ensure daily_picks is long enough
        #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')
        if len(daily_picks) > j and ((daily_picks[j] >= threshold) or np.isnan(daily_picks[j])):
            daily_stacked[j] = np.nan
            #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')

    
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
     
    inicio = data.index[0]
    final =  data.index[-1]
    
    plot_gpd = plot_GPD(data, picks, x, GPD, st, knee, threshold, inicio, final)
    plot2 = plot_detrend(idate, fdate, data, original_daily_stacked,daily_stacked, st, baseline_line)
###############################################################################
###############################################################################
#FILL GAPS BETWEEN EMPTY DAILY VALUES    
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
    return baseline_line#baseline_curve, undisturbed_days_sample

###############################################################################
#diurnal variation computation
###############################################################################
def get_diurnalvar(data, idx_daily, st):
    ndata = len(data)
    totdays = int(ndata/1440)
                   
    iqr_picks = max_IQR(data, 60, 24, method='stddev')    
    xaxis = np.linspace(1, 24, 1440)

    qd_baseline = []

#import UTC according to observatory
    ndays = 5
    info = night_time(net, st)
    utc = info[11]
    ini = 0
    fin = 0   
    
    try:
        utc = int(utc)  # Attempt to convert to an integer
    except ValueError:
        utc = float(utc)
    print(f"universal Coordinated time: {utc}") 
    
    qd_list = get_qd_dd(iqr_picks, idx_daily, 'qdl', ndays)
    #qd_list = ['2015-03-14', '2015-03-13', '2015-03-15', '2015-03-12']
    qdl = [[0] * 1440 for _ in range(ndays)]
    
    baseline = []
###############################################################################
#diurnal variation computation
###############################################################################
   # QDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']    
    print('qdl list, \t H[nT] \n')     
    print(qd_list)
    #plt.title('Local Quiet Days, June 2024: St: '+st, fontweight='bold', fontsize=18)
    for i in range(ndays):
        qd = (str(qd_list[i])[0:10])
        
        qd_arr = data[qd]
        
        qdl[i] = qd_arr
        
       # plt.plot(xaxis, qdl[i], label=f'QD{i+1}: {qd}')
        if utc <= 0:
            ini = int(abs(utc)*60)
            fin = ini+180    
            qd_2h = qdl[i].iloc[ini:fin]    
            baseline_value = np.nanmedian(qd_2h)
            baseline.append(baseline_value)
        elif utc >= 0:
            ini = int(1440 - abs(utc)*60)
            if (ini+180) <= 1440:
                fin = (ini + 180)
                qd_2h = qdl[i].iloc[ini:fin]
                baseline_value = np.nanmedian(qd_2h)
                baseline.append(baseline_value)       
            else:
                fin2 = (ini+180)-1440
                
                fin1 = ini + 59
                qd_2h = qdl[i].iloc[0:fin2]   
                qd_2h2 = qdl[i].iloc[ini:fin1]
                baseline_value1 = np.nanmedian(qd_2h)
                baseline_value2 = np.nanmedian(qd_2h2)
                baseline_value = (baseline_value1 + baseline_value2)/2
                baseline.append(baseline_value)
               
        baseline_value = np.nanmedian(qd_2h)
        baseline.append(baseline_value)
        qdl[i] = qdl[i] - baseline_value
        
        qdl[i] = qdl[i].reset_index()
        qdl[i] = qdl[i].drop(columns=['index'])

    # Generate the average array
# Combine all DataFrames into a single DataFrame with shape (n, 1440)
    qdl_concat = pd.concat(qdl, axis=1, ignore_index=True)
    
    # Compute the mean across rows (axis=0) to get a 1x1440 array
    qd_average = qdl_concat.mean(axis=1)        

    freqs = np.array([0.0, 1.1574e-5, 2.3148e-5, 3.4722e-5,4.6296e-5, \
                          5.787e-5, 6.9444e-5])    
    
    n = len(qd_average)
    N = len(qd_average)*totdays
    
    fs = 1/60
    f = fftpack.fftfreq(n, 1.0/fs)
    f = np.around(f, decimals = 9)
    mask = np.where(f >= 0)
    f=f[mask]
    
    fcomb = dcomb(n//2,1,f,freqs) 
    qd_average = np.array(qd_average)

 
    Gw = fftpack.fft(qd_average, axis=0)/np.sqrt(n)
    Gw = Gw[0:n//2]
    
    G_filt = Gw*fcomb.T 
    # Remove all zero comps in G_filt
    G = G_filt[G_filt != 0]                 # 1x7
    #G = np.matrix(G)                        
    k = np.pi/720
    
    td = np.arange(N).reshape(N,1)          # Nx1
    Td = np.kron(np.ones(7), td)            # Nx7
    
    phi = aphase(G)                                             # 1x7
    X = 2*abs(G)/np.sqrt(n)                                     # 1x7
    Ag = np.cos(k*np.multiply(Td, np.arange(7)) + phi)          # Nx7              
    ii = np.multiply(Ag,X)                                      # Nx7      
    suma = np.sum(ii, axis=1)                                   # Nx1  
    detrd = signal.detrend(suma)   
    T = np.median(np.c_[suma, detrd], axis=1)   
    
    template = T[0:1440]
    
    #plot_qdl(xaxis, template, ndays, qdl, st, idx_daily)
    qd_offset = np.nanmedian(baseline)

    return T, qd_offset
###############################################################################
###############################################################################
###############################################################################
#AUXILIAR FUNCTIONS 
def get_qd_dd(data, idx_daily, type_list, n):
    
    daily_var = {'Date': idx_daily, 'VarIndex': data}
    
    local_var = pd.DataFrame(data=daily_var)
    
    local_var = local_var.sort_values(by = "VarIndex", ignore_index=True)
    
    if type_list == 'qdl':
    
        local_var = local_var[0:n]['Date']   
    
    elif type_list == 'I_iqr':
    
        local_var = local_var.sort_values(by = "Date", ignore_index=True)
    
    return local_var
###############################################################################
#We call the base line derivation procedures
###############################################################################  
magdata = get_dataframe(filenames, path, idx, dates, net)

H = magdata['H']
X = magdata['X']
Y = magdata['Y']
Z = magdata['Z']
D = magdata['D']
I = magdata['I']

baseline_curve = base_line(H, idx, idx_daily)
base_lineX = base_line(X, idx, idx_daily)
base_lineY = base_line(Y, idx, idx_daily)
base_lineZ = base_line(Z, idx, idx_daily)

H_detrend = H-baseline_curve
X_detrend = X-base_lineX
Y_detrend = Y-base_lineY
Z_detrend = Z-base_lineZ
#diurnal base line
#sys.exit('end of the child process')
diurnal_baseline, offset = get_diurnalvar(H_detrend, idx_daily, st)
diurnal_baselineX, offsetX = get_diurnalvar(X_detrend, idx_daily, st)
diurnal_baselineY, offsetY = get_diurnalvar(Y_detrend, idx_daily, st)
diurnal_baselineZ, offsetZ = get_diurnalvar(Z_detrend, idx_daily, st)

H_raw = H

H = H_detrend-diurnal_baseline
X = X_detrend - diurnal_baselineX
Y = Y_detrend - diurnal_baselineY
Z = Z_detrend - diurnal_baselineZ

H_noff1 = H-offset
X_noff1 = X-offsetX
Y_noff1 = Y-offsetY
Z_noff1 = Z-offsetZ


dst = []
hr = int(len(H)/60)
for i in range(hr):
    tmp_h = np.nanmedian(H_noff1[i*60:(i+1)*60])
    dst.append(tmp_h)
    
#plot_process(H, H_raw, H_detrend, H_noff1, dst, baseline_curve, diurnal_baseline, st, idx_hr)

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
