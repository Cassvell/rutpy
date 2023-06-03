'''
En esta rutina tiene como propósito tomar un archivo de datos tec con una 
ventana de tiempo de varios días, cambiar su formato a una forma más legible, y 
retornar varios archivos de formato .txt, uno por cada día que abarque la 
ventana de estudio
'''
################################################################################
#importación de librerías
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import re
import os
################################################################################
#introducimos la ventana de tiempo que abarca el archivo original y que está 
#indicada en el nombre del mismo
idate = input("format code: (yyyy-mm-dd): ")
fdate = input("format code: (yyyy-mm-dd): ")

code='jpl' #código del archivo que hace referencia al modelo a partir del cual 
           #se derivaron los valores TEC
           
path='/home/isaac/geomstorm/datos/tec/tec_med/'
path2='/home/isaac/geomstorm/datos/tec/tec_locales/'
#revisión de si existe el archivo que se busca procesar, en función de la 
#ventana de tiempo y del modelo del que fue dervivado. 
if os.path.isfile(path+code+'_'+idate+'_'+fdate+'_med.txt'):
    code = 'jpl'
elif os.path.isfile(path+code+'_'+idate+'_'+fdate+'_med.txt'):
    code = 'codg' 
elif os.path.isfile(path+code+'_'+idate+'_'+fdate+'_med.txt'):
    code = 'nesp'       
elif os.path.isfile(path2+'tec_'+idate+'_'+fdate+'_med.txt'):
    code = 'nesp' 
   # path = path2          
else:
    print(' med file does not exist in the directory')

################################################################################
################################################################################
#generación del DataFrame de TEC esperado
df = pd.read_csv(path+code+'_'+idate+'_'+fdate+'_med.txt', header=0, delim_whitespace=True)
                 
                 
TEC_median = df.mean(axis=1) #se calcula el promedio de los 27 días previos a la
                             #ventana de tiempo de interés para generar el tec
                             #esperado o <TEC>                             
med_obs = TEC_median 
f = len(df.iloc[:,0])       #f = 12 si la frecuencia es 2H y 24 si es H.              
################################################################################
################################################################################
#generación del DataFrame de TEC observado
time_w = idate+'_'+fdate
name = path+code+'_'+time_w+'_tec.txt'

tec_df = pd.read_csv(name, header=0, \
                 delim_whitespace=True)
tec_df = tec_df[tec_df.columns.drop(list(tec_df.filter(regex=code+'g')))]
tec = pd.concat([tec_df, tec_df.T.stack().reset_index(name='tec')['tec']],\
                axis=1)
                

doy = tec_df.columns.tolist()
DOY_arr = np.asarray(doy)
DOY_arr = np.repeat(DOY_arr, len(tec_df))
DOY = pd.Series(DOY_arr)
tec['DOY'] = DOY.values
tec['DOY'] = (tec['DOY']).astype(float)
tec['DOY'] = (tec['DOY']).astype(int)
################################################################################
################################################################################
#generación del nuevo DataFrame con TEC y <TEC>
cols = tec.columns
tec = tec.drop(columns=cols[0:-2])
col = tec.columns.tolist()
col = col[-1:] + col[:-1]
tec = tec[col]                


resol = '2H'
if f == 12:
    ndays = len(DOY_arr)/12       #num de días que abarca la ventana de tiempo del archivo original
    enddata = fdate+ ' 22:00:00'
    resol = '2H'    
else:
    ndays = len(DOY_arr)/24
    enddata = fdate+ ' 23:00:00' 
    resol = 'H'                                           
med    = pd.concat([med_obs]*int(ndays), ignore_index=True) #expansión del 
median = pd.Series(med)                                     #arreglo <TEC> para    
                                                            #que coincida con la
tec['med'] = median.values                                  #dimensión de TEC
tec = tec.round(decimals=2)


tec['DOY'] = tec['DOY'].apply(lambda x: '{0:0>3}'.format(x)) #se rellena con 0's
tec['tec'] = tec['tec'].apply(lambda x: '{0:0>5}'.format(x)) #a la izq de los 
tec['med'] = tec['med'].apply(lambda x: '{0:0>5}'.format(x)) #val en cada col. 

################################################################################
################################################################################
#fragmentación del archivo original en varios archivos con un día de ventana de
#tiempo


#enddata = fdate+ ' 22:00:00'
#se genera un vector DateTime para poder fragmentar el DataFrame en función del
#tiempo
idx = pd.date_range(start = pd.Timestamp(idate), end = pd.Timestamp(enddata), \
                    freq=resol)

tec= tec.set_index(idx)
tec = tec.reset_index()

#en este bucle, se generan los archivos por cada día y se guardan en la dir 
#indicada.
for i in range(0,len(idx),12):
    mask = (tec['index'] >= idx[i]) & (tec['index'] <= idx[i+11])
    df = tec[mask]
    df = df.drop(columns=['index'])
    date = str(idx[i])
    date = date[0:10]                   #nuevo nombre en función del día de que  
    name_new = 'tec_'+date+'.txt'       #abarque cada archivo
    
    new_path = '/home/isaac/geomstorm/datos/tec/'
    df.to_csv(new_path+name_new, sep= ' ', index=False)  
    






