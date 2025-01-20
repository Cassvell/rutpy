'''
En esta rutina tiene como propósito tomar un archivo de datos tec con una 
ventana de tiempo de varios días, cambiar su formato a una forma más legible, y 
retornar varios archivos de formato .txt, uno por cada día que abarque la 
ventana de estudio. Es para leer y convertir los últimos archivos en .ods
'''
################################################################################
#importación de librerías
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import re
import os
import sys
################################################################################
#introducimos la ventana de tiempo que abarca el archivo original y que está 
#indicada en el nombre del mismo
print("input initial and final date: ")

idate = input("format code: (yyyy-mm-dd): ")
fdate = input("format code: (yyyy-mm-dd): ")

code='jpl' #código del archivo que hace referencia al modelo a partir del cual 
           #se derivaron los valores TEC
          
path='/home/isaac/MEGAsync/datos/tec/tec_locales/'
#revisión de si existe el archivo que se busca procesar, en función de la 
#ventana de tiempo y del modelo del que fue dervivado. 

################################################################################
################################################################################
'''
#generación del DataFrame de TEC esperado
df = pd.read_csv(path+code+'_'+idate+'_'+fdate+'_tec.txt', header=0, delim_whitespace=True)
                 
                 
TEC_median = df.mean(axis=1) #se calcula el promedio de los 27 días previos a la
                             #ventana de tiempo de interés para generar el tec
                             #esperado o <TEC>                             
med_obs = TEC_median 
f = len(df.iloc[:,0])       #f = 12 si la frecuencia es 2H y 24 si es H.        
'''      
################################################################################
################################################################################
#generación del DataFrame de TEC observado
time_w = idate+'_'+fdate
name = path+time_w+'.csv'

tec_df = pd.read_csv(name, header=0, sep='\t')
#print(tec_df)                 
f = len(tec_df.iloc[:,0])       #f = 12 si la frecuencia es 2H y 24 si es H. 
                 

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
#generacion del nuevo DataFrame con TEC y <TEC>
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
                                
tec = tec.round(decimals=2)

tec['DOY'] = tec['DOY'].apply(lambda x: '{0:0>3}'.format(x)) #se rellena con 0's
tec['tec'] = tec['tec'].apply(lambda x: '{0:0>5}'.format(x)) #a la izq de los 
################################################################################
################################################################################
#fragmentacion del archivo original en varios archivos con un día de ventana de
#tiempo


#enddata = fdate+ ' 22:00:00'
#se genera un vector DateTime para poder fragmentar el DataFrame en función del
#tiempo

year = str(idate[0:4])
TOTAL_DAYS = 0
if not int(year) % 4 and (int(year) % 100 or not int(year) % 400):
    TOTAL_DAYS = 366
else:
    TOTAL_DAYS = 365    

#YEAR = 2020
date = []
for i in tec['DOY']:
    tmp_date = datetime.strptime(year + "-" + str(i), "%Y-%j").strftime("%Y-%m-%d")    
    date.append(tmp_date)

tec['DATE']  = date

daily_day = [] 
for i in doy:
    tmp_date = datetime.strptime(year + "-" + str(i), "%Y-%j").strftime("%Y-%m-%d")    
    daily_day.append(tmp_date)

tec= tec.set_index(tec['DATE']).drop(columns=['DATE'])
#tec = tec.reset_index()

idx = pd.date_range(start = pd.Timestamp(idate), end = pd.Timestamp(fdate+' 22:00:00'), \
                    freq=resol)
#en este bucle, se generan los archivos por cada día y se guardan en la dir 
#indicada.

reference = pd.DataFrame(daily_day, doy)
print(reference)
print("######################################")
print('days to use for baseline: ')
print(reference.index[0])
print('to')
print(int(reference.index[0])+27)
if f == 12:
    for i in range(0,len(date),12):
        mask = (tec.index >= date[i]) & (tec.index <= date[i+11])
        df = tec[mask]
        df = df.reset_index()
        df = df.drop(columns=['DATE'])
     #   print(df) 
        name_new = 'tec_'+str(date[i])+'.dat'       #abarque cada archivo
     #   print(name_new) 
        new_path = '/home/isaac/MEGAsync/datos/tec/tec_newformat/'
        df.to_csv(new_path+name_new, sep= ' ', index=False)  
else:
    for i in range(0,len(date),24):
        mask = (tec['index'] >= date[i]) & (tec['index'] <= date[i+23])
        df = tec[mask]
        df = df.reset_index()
        df = df.drop(columns=['DATE'])
     #   print(df) 
        name_new = 'tec_'+str(date[i])+'.dat'       #abarque cada archivo
     #   print(name_new)       #abarque cada archivo
        
        new_path = '/home/isaac/MEGAsync/datos/tec/tec_newformat/'
        df.to_csv(new_path+name_new, sep= ' ', index=False)  
  
  
  
  
  
  
    
