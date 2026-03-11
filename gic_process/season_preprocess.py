import sys
import pandas as pd
import numpy as np
#from Ffitting import fit_data
import os
from timeit import default_timer as timer
from modules.window_27 import window_27
from gicdproc import  df_gic_pp
from modules.threshold import threshold
from modules.moving_window import hourly_IQR, max_IQR, med_IQR
from modules.diurnal_variation import diurnal_variation_model

###############################################################################
###############################################################################
#ARGUMENTOS DE ENTRADA
###############################################################################
idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

start = timer()
dirpath = '/home/isaac/datos/gics_obs/'
iwindows, med_windows, fwindows, nwindows= window_27(idate, fdate, 'date')

stat = ['LAV', 'QRO', 'RMY', 'MZT']

ndays=27
stat_dir = {}
#LEER TODOS LOS DATOS DISPONIBLES ENTRE FECHA INICIAL Y FECHA FINAL Y ALMACENARLOS EN UN DIRECTORIO
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

        if not np.all(np.isnan(window_data)):
            #CALCULAR PICOS DE VARIACION IQR POR CADA HORA, CON UNA TOLERANCIA DE 30% DE GAPS
            iqr_picks = hourly_IQR(window_data, 60, 0.7)
            threshold_gic = threshold(iqr_picks, iwindows[w], fwindows[w], st, '2s')
            sq_gic, monthly_baseline = diurnal_variation_model(window_data, idx_daily, 10, st.lower(), 'experimental', threshold_gic, 'gic')
            #corrected_data = window_data - sq_gic - monthly_baseline
            
            
            
        else:
            print(f'no data form {st} during the time window')
end = timer()
print(end - start)   
'''
PARA LA SEGUNDA PARTE, AGREGAR UNA SALIDA DE ARCHIVOS PROCESADOS DONDE SE LES ELIMINE LA VARIACION DIURNA, SE APLANEN LOS CAMBIOS DE OFFSET
EN EL CASO DE LAV Y SE CONSERVEN LOS THRESHOLDS POR CADA VENTANA DE TIEMPO

'''