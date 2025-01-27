import pandas as pd
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from scipy.stats import genpareto, kstest #anderson
#from datetime import datetime
from scipy.optimize import curve_fit 
# Ajuste de distribuciones
import sys
from lmoments3 import distr
import lmoments3 as lm
import kneed as kn
from numpy.linalg import LinAlgError
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import NearestNDInterpolator
from magnetic_datstruct import get_dataframe
from scipy.signal import medfilt
from aux_time_DF import index_gen, convert_date
from scipy.fft import fft, ifft
from scipy import fftpack
from scipy import signal
from typical_vall import night_hours, mode_nighttime, typical_value, gaus_center, mode_hourly
import cmath
from Ffitting import fit_data
from night_time import night_time

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
                    end = pd.Timestamp(enddata), freq='T')
idx_hr = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='H')    
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')
filenames = []
filenames_out = []
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
        filenames.append(fname)
        filenames_out.append(fname2)
else:
    year = '2015'
    st_dir = st.upper()
    path = f"/home/isaac/datos/{net}/{year}/{st_dir}/" # magnetic data path
    for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        fname = st+date_name_newf+'dmin.min'
        #print(fname)
        fname2 = st+'_'+date_name+'.dat'
        filenames.append(fname)
        filenames_out.append(fname2)
     
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
    ndays = int(ndata/1440)

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
    disturbed_days_sample = []
    undisturbed_days_sample = []
    line_styles = ['-', '--', '-.', ':']
    '''  
    for i, window in enumerate(pickwindow):
        # Compute threshold and GPD
        picks = max_IQR(data, 60, window, method='iqr')
        
        x, GPD, threshold = get_threshold(picks)

        # Validate GPD fit using the second derivative
        second_derivative = np.gradient(np.gradient(GPD))
        if np.all(second_derivative >= 0):
            break

        # Daily IQR picks and classification
        daily_picks = max_IQR(data, 60, 24, method='iqr')
        print(daily_picks)
        i_iqr = get_qd_dd(daily_picks, idx_daily, 'I_iqr', ndays)['VarIndex']

        # Reset daily_stacked and classify days

        is_disturbed = i_iqr >= threshold
        daily_stacked[is_disturbed] = np.nan
        
        
        undisturbed_days = daily_stacked[~np.isnan(daily_stacked)]
        trials = len(undisturbed_days)
        
        # Plot results
        style = line_styles[i % len(line_styles)]
        plt.hist(picks, density=True, bins=ndays * 2, histtype='stepfilled', alpha=0.6)
        plt.plot(x, GPD, lw=2, label=f'Window: {window} hr')
        plt.axvline(x=threshold, color='k', linestyle=style, label=f'Threshold: {threshold:.2f}')
        plt.legend()
        plt.show()
    '''  
    picks = max_IQR(data, 60, pickwindow[0], method='iqr')
    
    x, GPD, threshold = get_threshold(picks)

    # Validate GPD fit using the second derivative
    second_derivative = np.gradient(np.gradient(GPD))

    # Daily IQR picks and classification
    daily_picks = max_IQR(data, 60, 24, method='iqr')

    for j in range(len(daily_stacked)):
        # Ensure daily_picks is long enough
        #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')
        if len(daily_picks) > j and ((daily_picks[j] >= threshold) or np.isnan(daily_picks[j])):
            daily_stacked[j] = np.nan
            #print(f'fecha: {idx_daily[j]}, valor diario: {daily_stacked[j]}, iqr max: {daily_picks[j]}')

    
    
    
    # Plot results
    style = line_styles[i % len(line_styles)]
    plt.title('TEO OBS')
    plt.hist(picks, density=True, bins=ndays * 2, histtype='stepfilled', alpha=0.6)
    plt.plot(x, GPD, lw=2, color='r', label=f'Window: {pickwindow[0]} hr')
    plt.axvline(x=threshold, color='k', linestyle=style[0], label=f'Threshold: {threshold:.2f}')
    plt.ylabel('Probabilidad')
    plt.xlabel('Picos de variación IQR [bin: 3 h]')
    plt.legend()
    plt.show()
    baseline_line = [np.nanmedian(daily_stacked)]*ndata
    idx_daily2 = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')+ pd.DateOffset(hours=6)

    fig, ax = plt.subplots(4, figsize=(12,8), dpi = 300) 
    fig.suptitle(st+' Geomagnetic Obs' , fontsize=24, \
                fontweight='bold') 
    inicio = data.index[0]
    final =  data.index[-1]
    
    ax[0].plot(data.index, data, label='raw data')
    ax[0].plot(idx_daily2, original_daily_stacked, 'ro', label='<datos nocturnos>')
    #ax[0].axhline(y = baseline_line[0], color='g', label='base line monthly tendency')
    ax[0].grid()
    ax[0].set_xlim(inicio,final)
    ax[0].set_ylabel('BH [nT]', fontweight='bold')
    ax[0].legend()

    ax[1].plot(data.index, data, label='raw data')
    ax[1].plot(idx_daily2, daily_stacked, 'ro', label='<datos nocturnos filtrados>')
    #ax[0].axhline(y = baseline_line[0], color='g', label='base line monthly tendency')
    ax[1].grid()
    ax[1].set_xlim(inicio,final)
    ax[1].set_ylabel('BH [nT]', fontweight='bold')
    ax[1].legend()



    ax[2].plot(data.index, baseline_line, color='r', label='monthly baseline')
    ax[2].plot(data.index, data, label='raw data')
    ax[2].grid()
    ax[2].set_xlim(inicio,final)
    ax[2].set_ylabel('BH [nT]', fontweight='bold')
    ax[2].legend()

    ax[3].plot(data.index, data - baseline_line, label='H - H0')
    ax[3].grid()
    ax[3].set_xlim(inicio,final)
    ax[3].set_ylabel('BH [nT]', fontweight='bold')
    ax[3].legend()


    fig.savefig("/home/isaac/MEGAsync/posgrado/doctorado/semestre4/procesado/"+\
                st+'_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
    plt.tight_layout() 
    plt.show()
  
###############################################################################
###############################################################################
#FILL GAPS BETWEEN EMPTY DAILY VALUES    
    
    return baseline_line#baseline_curve, undisturbed_days_sample

###############################################################################
#diurnal variation computation
###############################################################################
def get_diurnalvar(data, idx_daily, st):
    ndata = len(data)
    ndays = int(ndata/1440)
                   
    iqr_picks = max_IQR(data, 60, 24, method='stddev')    
    xaxis = np.linspace(1, 24, 1440)

    qd_baseline = []

#import UTC according to observatory
    n = 4
    info = night_time(net, st)
    utc = info[11]
    ini = 0
    fin = 0   
    
    try:
        utc = int(utc)  # Attempt to convert to an integer
    except ValueError:
        utc = float(utc)
    print(f"universal Coordinated time: {utc}") 
    
    qd_list = get_qd_dd(iqr_picks, idx_daily, 'qdl', n)
    #qd_list = ['2015-03-14', '2015-03-13', '2015-03-15', '2015-03-12']
    qdl = [[0] * 1440 for _ in range(n)]
    
    baseline = []
###############################################################################
#diurnal variation computation
###############################################################################
   # QDS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']    
    print('qdl list, \t H[nT] \n')     
    print(qd_list)
    #plt.title('Local Quiet Days, June 2024: St: '+st, fontweight='bold', fontsize=18)
    for i in range(n):
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
        
        print(f"index inicial: {ini}, index final: {fin}")
               
        baseline_value = np.nanmedian(qd_2h)
        baseline.append(baseline_value)
        qdl[i] = qdl[i] - baseline_value
        
        qdl[i] = qdl[i].reset_index()
        qdl[i] = qdl[i].drop(columns=['index'])
        plt.plot(xaxis, qdl[i])
        #print(qdl[i])     
        #print(qd, ' | ',  max(qdl[i]))
    #sys.exit('Exiting child process') 
    # Generate the average array
# Combine all DataFrames into a single DataFrame with shape (n, 1440)
    qdl_concat = pd.concat(qdl, axis=1, ignore_index=True)
    
    # Compute the mean across rows (axis=0) to get a 1x1440 array
    qd_average = qdl_concat.mean(axis=1)        

    # Optional: Plot the averaged/median data
    #plt.plot(xaxis, qd_average, label="Average QDL",  color='k',linewidth=4.0 )
    #plt.legend()
    #plt.show()
    #sys.exit("Exiting the code with sys.exit()!")
    freqs = np.array([0.0, 1.1574e-5, 2.3148e-5, 3.4722e-5,4.6296e-5, \
                          5.787e-5, 6.9444e-5])    
    
    n = len(qd_average)
    N = len(qd_average)*ndays
    
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

    plt.plot(xaxis, T[0:1440], label="model", color='k',linewidth=4.0 )
    plt.legend()
    plt.show()
    
###############################################################################
    path = '/home/isaac/MEGAsync/posgrado/doctorado/images/qdl/'
    diurnal_baseline = np.tile(qd_average, ndays) 
    #plt.plot(xaxis, qd_baseline, label="<QDL>", color='k',linewidth=4.0 )
    #plt.plot(xaxis, QD_baseline_min, color='b', linewidth=4.0, label = 'model Fit')
    #plt.xlim(1.0,24.0)
    #plt.xlabel('Time [UTC]', fontweight='bold', fontsize=16)
    #plt.ylabel('H [nT]', fontweight='bold', fontsize=16)    
    #plt.legend()
    #plt.savefig(path+'june'+'_'+st, dpi=200)
    qd_offset = np.nanmedian(baseline)
    #plt.show()
    return T, qd_offset
###############################################################################
###############################################################################
###############################################################################
#AUXILIAR FUNCTIONS 
###############################################################################
###############################################################################
#COMPUTATION OF THE THRESHOLD
###############################################################################
###############################################################################
###############################################################################
def aphase(array):
    
    """ Calcula la fase geometrica (en radianes) de todos los elementos de 
    un array 1D complejo"""
    
    output=[]
    for j in range(len(array)):
        pp = cmath.phase(array[j])
        output.append(pp)
        
    output = np.array(output)
    return output 

def dcomb(n, m, f, freqs):
    
    """ Crea un peine de Dirac de dimensions [n,m], para un conjunto de 
    frecuencias predeterminadas en un vector o lista de frecuencias dadas """
    
    delta = np.zeros([n, m])    
    for w in freqs:    
        idx = np.where(abs(f) == w)
        uimp = signal.unit_impulse([n,m], idx)
        delta = delta + uimp 
    
    return delta

def get_threshold(picks):

    ndays = int(len(picks)/4)

    picks = np.array(picks)  

    picks = picks[~np.isnan(picks)]
    
    hist, bins = np.histogram(picks, bins=ndays*2, density=True)  
 
    GPD_paramet = distr.gpa.lmom_fit(picks)

    shape = GPD_paramet['c']
    
    threshold = GPD_paramet['loc']
    
    scale = GPD_paramet['scale']
    
    x = np.linspace(min(picks), max(picks), len(picks))    
    
    GPD =  genpareto.pdf(x, shape, loc=threshold, scale=scale)
    
    GPD = np.array(GPD)
    
    if any(v == 0.0 for v in GPD):
        GPD =  genpareto.pdf(x, shape, loc=min(bins), scale=scale)
   
    params = genpareto.fit(picks)
    D, p_value = kstest(picks, 'genpareto', args=params)
    print(f"K-S test result:\nD statistic: {D}\np-value: {p_value}")
    
# Interpretation of the p-value & TEST KS for evaluating IQR picks
    alpha = 0.05

    if p_value > alpha:
        print("Fail to reject the null hypothesis: data follows the GPD")
    else:
        print("Reject the null hypothesis: data does not follow the GPD")   
        
    kneedle = kn.KneeLocator(
        x,
        GPD,
        curve='convex',
        direction='decreasing',
        S=5,
        online=True,
        interp_method='interp1d',
    )

    knee_point = kneedle.knee #elbow_point = kneedle.elbow

    print(f'knee point: {knee_point}')

#The Knee point is then considered as threshold.

    return x, GPD, knee_point
###############################################################################
###############################################################################
#generates an array of variation picks    
import numpy as np
import sys

def med_IQR(data, tw, tw_pick, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()

    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / tw)
        hourly = []
        
        for i in range(hourly_sample):
            current_window = data[i * tw : (i + 1) * tw]
            
            if len(current_window) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
            
            if non_nan_ratio > 0.9:
                QR1_hr = np.nanquantile(current_window, 0.25)
                QR3_hr = np.nanquantile(current_window, 0.75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            
            hourly.append(iqr_hr)
        
        return hourly
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanstd(trihourly_window)  # Standard deviation of the window
        else:
            stdev = np.nan  # If the window is incomplete (less than 3 data points)
        trihourly_stdev.append(stdev)

    
    daily = []
    
    # For each day, we pick the maximum IQR or standard deviation based on tw_pick
    for i in range(int(24 / tw_pick) * ndays):        
        if method == 'iqr':
            iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.90:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.9:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)

def max_IQR(data, tw, tw_pick, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()

    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / tw)
        hourly = []
        
        for i in range(hourly_sample):
            current_window = data[i * tw : (i + 1) * tw]
            
            if len(current_window) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
            
            if non_nan_ratio > 0.9:
                QR1_hr = np.nanquantile(current_window, 0.25)
                QR3_hr = np.nanquantile(current_window, 0.75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            
            hourly.append(iqr_hr)
        
        return hourly
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanstd(trihourly_window)  # Standard deviation of the window
        else:
            stdev = np.nan  # If the window is incomplete (less than 3 data points)
        trihourly_stdev.append(stdev)

    
    daily = []
    
    # For each day, we pick the maximum IQR or standard deviation based on tw_pick
    for i in range(int(24 / tw_pick) * ndays):        
        if method == 'iqr':
            iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.90:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.9:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)
###############################################################################
#based on IQR picks index, select either the 5 QDL in date yyyy-mm-dd format
#in case of type_list = 'qdl' if type_list = I_iqr, it returns a list of the 
#IQR picks per day
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

def mz_score(x):
    
    """ Modified z-score to identify outliers
    If MAD != 0 uses 0.6745 which is the value of the 3rd quantile in the normal
    distribution of probability.
    If MAD = 0 we approximate through the meanAD with 0.7979 the ratio between meanAD
    to the std deviation for the normal distribution

    """
    
    median_int = np.nanmedian(x)
    mad_int = np.nanmedian(np.abs(x - median_int))
    if mad_int <= 1e-15 :
        mean_int = np.nanmean(x)
        mean_ad_int = np.nanmean(np.abs(x - mean_int))
        mz_scores = 0.7979 * (x - median_int) / mean_ad_int 
    else:     
        mz_scores = 0.6745 * (x - median_int) / mad_int
        
    return mz_scores
def fixer(y, m=7, threshd = 7.5):
    
    """ Wittaker-Hayes  Algorithm to identify outliers in a time series """   
    # thereshold: binarization threshold. 
    
    yp = np.pad(y, (m,m+1), 'mean')
    delta = np.diff(yp, axis=0)
    spikes = np.abs(mz_score(delta)) >= threshd  #n-1
    y_out = yp.copy()                     # So we don’t o verwrite y
    for i in np.arange(len(spikes)-m-1):
        if spikes[i] != 0:               # If we have an spike in position i
            w = np.arange(i-m,i+m+1)     # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0]
            #w3 = w[spikes[w] != 0]
            # From such interval, we choose the ones which are not spikes
            y_out[i] = np.nanmedian(yp[w2])  # and we take the median of their values
            #y_out[i] = np.interp(i, w2, yp[w2])
            
    return y_out[m:len(delta)-m]   



    
def despike(y, threshd = 7.5):
    
    """ Search and replace spikes in an array with NaNs  """
       # thereshold: binarization threshold. 
       
    yp = np.pad(y, (0,1))
    delta = np.diff(yp, axis=0)
    spikes = np.abs(mz_score(delta)) >= threshd  #n
    #y_out = yp.copy()                     # So we don’t o verwrite y
    y_out = np.where(spikes !=0, np.nan, y)
    return y_out 
H = get_dataframe(filenames, path, idx, dates, net)

H = despike(H, threshd = 7.5)

for i in range(len(H)):
    if H[i] > 60000:
        H[i] = np.nan
        

df_H = pd.DataFrame(H)
df_H = df_H.set_index(idx)

H = df_H.iloc[:,0]
H_raw = H
baseline_curve = base_line(H, idx, idx_daily)

H_detrend = H-baseline_curve
sys.exit('end of child process')
#diurnal base line
diurnal_baseline, offset = get_diurnalvar(H_detrend, idx_daily, st)

H = H_detrend-diurnal_baseline

H_noff1 = H-offset

dst = []
hr = int(len(H)/60)
for i in range(hr):
    tmp_h = np.nanmedian(H_noff1[i*60:(i+1)*60])
    dst.append(tmp_h)
    

fig, ax = plt.subplots(4, figsize=(12,8), dpi = 300) 
fig.suptitle(st+' Geomagnetic Obs' , fontsize=24, \
             fontweight='bold') 
inicio = H.index[0]
final =  H.index[-1]
ax[0].plot(H.index, H_raw, label='raw data')
ax[0].plot(H.index, baseline_curve, color='r', label='monthly baseline')
#ax[0].axhline(y = baseline_line[0], color='g', label='base line monthly tendency')
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_ylabel('BH [nT]', fontweight='bold')
ax[0].legend()


ax[1].plot(H.index, H_detrend, label='H - base curve')
ax[1].plot(H.index, diurnal_baseline, color='r', label='diurnal variation')
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_ylabel('BH [nT]', fontweight='bold')
ax[1].legend()

ax[2].plot(H.index, H_noff1, color='k', \
           label='H - (diurnal baseline + baseline+offset)')
ax[2].grid()
ax[2].set_xlim(inicio,final)
ax[2].set_ylabel(' BH [nT]', fontweight='bold')
ax[2].legend()

ax[3].plot(idx_hr, dst, color='k', label='Dst')
ax[3].grid()
ax[3].set_xlim(inicio,final)
ax[3].set_ylabel(' BH [nT]', fontweight='bold')
ax[3].legend()
fig.savefig("/home/isaac/MEGAsync/posgrado/doctorado/semestre4/procesado/"+\
            st+'_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.tight_layout() 
plt.show()

dat = {'H' : H_noff1, 'baseline_line' : baseline_curve, \
       'SQ' : diurnal_baseline   }

import os
  
df = pd.DataFrame(dat)  
path =  f"/home/isaac/datos/{net}/{st}/minV2/"  
isExist = os.path.exists(path)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")    


for i in range(len(idx_daily)):
    #print(df['H'][i*1440:((i+1)*1440)-1 ])
    daily_data = df[i*1440:((i+1)*1440) ]
    daily_data.reset_index(drop=True, inplace=True)
    # Replace NaN values with 99999.9
    daily_data.fillna(9999.9, inplace=True)
    full_path = os.path.join(path, filenames_out[i])
    filenames.append(full_path)
    # Prepare the file for manual formatting
    with open(full_path, 'w') as f:
        for index, row in daily_data.iterrows():
            # Manual formatting with fixed-width or specific spacing (e.g., two columns of F10.2)
            line = f"{row['H']:7.2f}{row['baseline_line']:10.2f}{row['SQ']:6.2f} \n"  # Adjust spacing as needed
            f.write(line)
    print(f"Saved: {full_path}")
#df.to_csv()
