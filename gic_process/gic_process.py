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

stat  = ['LAV', 'QRO', 'RMY', 'MZT']
dir_path = f'/home/isaac/datos/gics_obs/'
fig_path = '/home/isaac/gics_rv/fig/processed/'
#PRIMERA COLUMNA: MODELO DE VARIACION DIURNA
#SEGUNDA COLUMNA: VARIACION HORA A HORA
#TERCERA COLUMNA: LINEA BASE DE LA VENTANA DE TIEMPO
stat_dir_sq = {}
stat_dir = {}
amp_dir = {}
baselines = {}

for st in stat:
    #print(f'{st}')
    window_data = gic_qd(idate, fdate, f'{dir_path}qdl/', st, 'gic')
    window_data = window_data.replace(999.9, np.nan)
    stat_dir_sq[st] = window_data
    

for st in stat:
    data = df_gic_pp(idate, fdate, dir_path, st)
    gic = data['gic']
    stat_dir[st] = gic


for w in range(nwindows):
    print(f'Window {iwindows[w]} to {fwindows[w]}:')
    idx_daily = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(fwindows[w]), freq='D')
    idx_min = pd.date_range(start = pd.Timestamp(iwindows[w]),  end = pd.Timestamp(fwindows[w]), freq='min')
    cleaned_data = {}
    
    fig, axes = plt.subplots(4, 1, figsize=(25, 20))
    colors = ['blue', 'orange', 'green', 'purple']
    for (idx, st)  in enumerate(stat):
        print(f'{st}')
        
        #DIVIDIR LOS DATOS DE LAS 4 ESTACIONES EN SEGMENTOS DE 27 DIAS  
        window_data = stat_dir[st][iwindows[w]:fwindows[w]]
        window_tmp = stat_dir_sq[st][w*1440:(w+1)*1440]
        
        
        
        sq_gic = window_tmp.iloc[:,0]
        #print(sq_gic)
        baseline = window_tmp.iloc[:,2]
        
        sq_gic_iterated = np.tile(sq_gic, 27)
        baseline_iterated = np.tile(baseline, 27)
        
        #print(len(window_data), len(window_tmp))
        if not np.all(np.isnan(window_data)):      
            iqr_picks = hourly_IQR(window_data, 60, 0.7)
            if not np.all(np.isnan(iqr_picks)):      
                threshold_gic = threshold(iqr_picks,iwindows[w], fwindows[w], st, '2s')
            
                      
            offset_idx = detect_offset(window_data, 10, 120, threshold_gic/2)
            cleaned_data = window_data-sq_gic_iterated-baseline_iterated
            
            if not len(offset_idx) == 0:
                threshold_jump = 10-threshold_gic/2
                corrected_data = corr_offset(cleaned_data, offset_idx, threshold_jump)
            else:    
                corrected_data = cleaned_data 
             
            processed_data = pd.DataFrame(corrected_data).set_index(idx_min)            
        #sq_gic = stat_dir_sq[st][iwindows[w]:fwindows[w]]
        else:
            #print(f'no data form {st} during the time window')
            tmp_data = (np.full(27*1440,np.nan))
            processed_data = pd.DataFrame(tmp_data).set_index(idx_min)
        

        
        cleaned_data[st] = processed_data

     

        for i in range(27):
            gic_processed = pd.DataFrame(processed_data[i*1440:(i+1)*1440]).reset_index()
            gic_processed.columns = ['DateTime', 'gics']
            gic_processed.fillna(999.9, inplace=True)
            date_str = gic_processed['DateTime'][0].strftime('%Y%m%d')
            year = gic_processed['DateTime'][0].strftime('%Y')
            output_path = f'{dir_path}processed/{year}/{st}/'
            directory = os.path.dirname(output_path)
            if not os.path.exists(output_path):
                    os.makedirs(output_path, exist_ok=True)           
            
            filename = f'{st}_{date_str}.p.csv'
            full_path = os.path.join(output_path, filename)
            gic_processed.to_csv(full_path, index=False, sep='\t')
 
        axes[idx].plot(idx_min, window_data, alpha=0.6, color='k' , label=f'Original', linewidth=1)
        axes[idx].plot(idx_min, cleaned_data[st], color=colors[idx], linewidth=2, label=f'Corregido')

        # Configuraciones del subplot
        axes[idx].set_ylabel(f'{st} - GICS [A]', fontsize=22)
        axes[idx].legend(loc='upper right', fontsize=22)
        axes[idx].grid(True, alpha=0.3)

        axes[idx].set_xlim(idx_min[0], idx_min[-1])
        axes[idx].xaxis.set_major_locator(plt.MaxNLocator(10))        

        axes[idx].tick_params(axis='x', labelsize=22)  
        axes[idx].tick_params(axis='y', labelsize=22)  

    ini_w = iwindows[w].strftime('%Y-%m-%d')
    fin_w = fwindows[w].strftime('%Y-%m-%d')

    
    plt.tight_layout()
    
    path_figure = f'{fig_path}{ini_w}_{fin_w}.png'
    directory = os.path.dirname(path_figure)
    if directory and not os.path.exists(path_figure):
        os.makedirs(directory, exist_ok=True)
    
    plt.savefig(f'{path_figure}', dpi=300)
    plt.close()


        
        #cleaned_data[st] = corrected_data

#GENERAR Y GUARDAR FIGURAS DONDE SE SOBREPONGAN LOS DATOS GICS ANTES Y DESPUES DE REMOVER RUIDO. SERIVIRA USAR UN ALPHA=0.6 PARA
#LOS DATOS ORIGINALES. DEBE TENER 4 SUBPLOTS. TAMBIÉN ES NECESARIO GENERAR UN DIRECTORIO DONDE SE GUARDARAN LAS FIGURAS PROCESADAS
end = timer()
print(end - start) 



        