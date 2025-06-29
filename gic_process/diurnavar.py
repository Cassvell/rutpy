import numpy as np
import pandas as pd
from scipy import fftpack, signal
from lowpass_filter import aphase, dcomb
from plots import plot_GPD, plot_detrend, plot_qdl ,plot_process
from threshold import max_IQR


def get_diurnalvar(data, idx_daily, st):
    ndata = len(data)
    totdays = int(ndata/1440)
                   
    iqr_picks = max_IQR(data, 60, 24, method='stddev')    
    xaxis = np.linspace(1, 24, 1440)

    qd_baseline = []

#import UTC according to observatory
    ndays = 5
    #info = night_time(net, st)
    utc = -6
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
    
    plot_qdl(xaxis, template, ndays, qdl, st, idx_daily)
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