import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from modules.corr_offset import corr_offset, detect_offset
import os
from timeit import default_timer as timer
from modules.window_27 import window_27
from gicdproc import  df_gic_pp, gic_qd
from modules.threshold import threshold
from modules.moving_window import hourly_IQR
#from modules.threshold import threshold

idate = sys.argv[1]
fdate = sys.argv[2]

fyear = int(fdate[0:4])
fmonth = int(fdate[4:6])
fday = int(fdate[6:8])

stat = ['LAV', 'QRO', 'RMY', 'MZT']
#stat = ['MZT', 'QRO', 'RMY', 'MZT']
#st = ['QRO', 'QRO', 'RMY', 'MZT']
path = f'/home/isaac/datos/gics_obs/'





###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

start = timer()

iwindows, med_windows, fwindows, nwindows= window_27(idate, fdate, 'date')

dirpath = '/home/isaac/datos/gics_obs/'
stat  = ['LAV', 'QRO', 'RMY', 'MZT']
dir_path = f'/home/isaac/datos/gics_obs/qdl/'

#PRIMERA COLUMNA: MODELO DE VARIACION DIURNA
#SEGUNDA COLUMNA: VARIACION HORA A HORA
#TERCERA COLUMNA: LINEA BASE DE LA VENTANA DE TIEMPO
stat_dir_sq = {}
stat_dir = {}
amp_dir = {}
baselines = {}

for st in stat:
    #print(f'{st}')
    window_data = gic_qd(idate, fdate, dir_path, st, 'gic')
    window_data = window_data.replace(999.9, np.nan)
    stat_dir_sq[st] = window_data

for st in stat:
    data = df_gic_pp(idate, fdate, dirpath, st)
    gic = data['gic']
    stat_dir[st] = gic

for w in range(nwindows):
    print(f'Window {iwindows[w]} to {fwindows[w]}:')
    idx_daily = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(fwindows[w]), freq='D')
    for st in stat_dir:
        print(f'{st}')
        
        #DIVIDIR LOS DATOS DE LAS 4 ESTACIONES EN SEGMENTOS DE 27 DIAS  
        window_data = stat_dir[st][iwindows[w]:fwindows[w]]
        window_tmp = stat_dir_sq[st][w*1440:(w+1)*1440]
        
        sq_gic = window_tmp.iloc[:,0]
        baseline = window_tmp.iloc[:,2]
        
        sq_gic_iterated = np.tile(sq_gic, 27)
        baseline_iterated = np.tile(baseline, 27)
        
        
        
        if not np.all(np.isnan(window_data)):      
            iqr_picks = hourly_IQR(window_data, 60, 0.7)

            threshold_gic = threshold(iqr_picks,iwindows[w], fwindows[w], st, '2s')
            
            offset_idx = detect_offset(window_data, 10, 120, threshold_gic/2)

            if not len(offset_idx) == 0:
                corrected_data = corr_offset(window_data, offset_idx, 10-threshold_gic/2)
                cleaned_data = corrected_data-sq_gic_iterated-baseline_iterated    
            else:    
                cleaned_data = window_data-sq_gic_iterated-baseline_iterated
        
        plt.plot(window_data.index, sq_gic_iterated)
        plt.plot(window_data, 'k')
        plt.plot(window_data.index, window_data-sq_gic_iterated-baseline_iterated, 'r', linewidth=3)
        plt.show()
        #sq_gic = stat_dir_sq[st][iwindows[w]:fwindows[w]]
        if not np.all(np.isnan(window_data)):
            #CALCULAR PICOS DE VARIACION IQR POR CADA HORA, CON UNA TOLERANCIA DE 30% DE GAPS 
            d = 0
            #corrected_data = window_data - sq_gic - monthly_baseline
            
            
            
        else:
            print(f'no data form {st} during the time window')

   
end = timer()
print(end - start) 

sys.exit('termina HDTPM!!!')
        
        #plt.plot(gic_res, label=f'{i} GIC no Diurnal Base', alpha=0.7)
fig, axes = plt.subplots(4, 2, figsize=(25, 20))
colors = ['blue', 'orange', 'green', 'purple']

# Processed + QD Model in left column (0, 1, 2, 3)
for i, station in enumerate(dict_gic):
    axes[i, 0].plot(pp_gic[station], label=f'{station} Processed', color=colors[i], alpha=0.7, linewidth=1.5)
    axes[i, 0].plot(idx1, dict_qd[station], label=f'{station} QD Model', color='red', alpha=0.7, linewidth=1.5)
    axes[i, 0].set_title(f'{station} - Processed GIC vs QD Model')
    axes[i, 0].set_ylabel('GIC')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)

# Raw data in right column (0, 1, 2, 3)
for i, station in enumerate(dict_gic):
    axes[i, 1].plot(dict_gic[station], label=f'{station} no QD', color=colors[i], alpha=0.7, linewidth=1.5)
    axes[i, 1].set_title(f'{station} - Raw GIC Data')
    axes[i, 1].set_ylabel('GIC')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'/home/isaac/rutpy/processed/gic_processed_{idate}_{fdate}.png', dpi=300)
plt.close()
#plt.show()
sys.exit('end of child process')
output_path = f'/home/isaac/datos/gics_obs/processed/{fyear}/'

for i in stat:
    if not os.path.exists(output_path + i):
        os.makedirs(output_path + i)

    ndays = int(len((dict_gic[i]))/1440)
    for j in range(ndays):
        start_idx = j * 1440
        end_idx = (j + 1) * 1440
        daily_data = dict_gic[i].iloc[start_idx:end_idx]
        daily_data.fillna(999.9, inplace=True)
        date_str = daily_data.index[0].strftime('%Y%m%d')
        daily_data.to_csv(output_path + f'{i}/gic_{i}_{date_str}.csv', header=False)
    #for j in 
    #dict_gic[i].to_csv(output_path + f'{i}/gic_{i}_{idate}_{fdate}.csv', header=True)


        