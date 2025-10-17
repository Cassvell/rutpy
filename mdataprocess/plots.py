import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
if os.path.exists('/home/isaac/MEGAsync/posgrado/doctorado/'):
    path = '/home/isaac/MEGAsync/posgrado/doctorado/'
else:
    path = '/home/isaac/MEGA/posgrado/doctorado/'

def plot_GPD(data, bindat, x, GPD, st, knee, threshold, inicio, final):
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
# Plot results         
    picks = bindat[~np.isnan(bindat)]
    picks = np.unique(bindat)
    
    sorted_picks = np.sort(picks)
    reversed_picks = sorted_picks[::-1]  
    ndays = int(len(data)/1440)
    plt.title(f'{st}')
    plt.hist(reversed_picks, density=True, bins=ndays * 2, histtype='stepfilled', alpha=0.6, label='sorted peak values')
    #plt.hist(bindat, density=True, bins=ndays * 2, histtype='stepfilled', alpha=0.4, label='peak values')
    plt.plot(x, GPD, lw=2, color='r', label=f'fitted GPD')
    plt.axvline(x=knee, color='k', linestyle='--', label=f'Knee point: {knee:.2f}')    
    plt.axvline(x=threshold, color='k', linestyle='-', label=f'Threshold: {threshold:.2f}')
    plt.ylabel('Probabilidad')
    plt.xlabel('Picos de variación IQR [bin: 3 h]')
    plt.legend()
    plt.savefig(f"{path}semestre4/"+\
                st+'_'+str(inicio)[0:10]+"_"+str(final)[0:10]+"_GPD.png")
    plt.tight_layout() 
    plt.show()
    return 


def plot_detrend(idate, enddata, data, original_daily_stacked,daily_stacked, st, baseline_line):
    idx_daily2 = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')+ pd.DateOffset(hours=6)        
    
    fig, ax = plt.subplots(4, figsize=(12,8), dpi = 300) 
    fig.suptitle(st+' Geomagnetic Obs' , fontsize=24, \
                fontweight='bold') 
    inicio = data.index[0]
    final =  data.index[-1]
    
    ax[0].plot(data.index, data, label='raw data')
    ax[0].plot(idx_daily2, original_daily_stacked, 'ro', label='<datos nocturnos>')
   # ax[0].axhline(y = baseline_line[0], color='g', label='base line monthly tendency')
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


    fig.savefig(f"{path}semestre4/"+\
                st+'_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
    plt.tight_layout() 
    plt.show()
    return 

def plot_qdl(xaxis, template, n, qdl, st, idx_daily):
    inicio = idx_daily[0].strftime('%Y-%m-%d')
    final = idx_daily[-1].strftime('%Y-%m-%d')
    print(n)
    for i in range(n):
        if len(qdl[i]) == 1440:
            plt.plot(xaxis, qdl[i])
    
    plt.plot(xaxis, template, label="model", color='k',linewidth=4.0 )
    plt.legend()
    plt.title(f'{st.upper()} diurnal variation')   
    plt.tight_layout() 
 
    plt.savefig(f'{path}semestre5/qdl/{st}_{inicio}_{final}.png')

    plt.show()
    return

def plot_process(H, H_raw, H_detrend, H_noff1, dst, baseline_curve, diurnal_baseline, st, idx_hr):
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
    fig.savefig(f"{path}semestre4/"+\
                st+'_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
    plt.tight_layout() 
    plt.show()