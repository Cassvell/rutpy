import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

from PIL import Image
from datetime import datetime, timedelta
import glob, os
import fnmatch 

from os import listdir
from os.path import isfile, join

import itertools

from scipy import signal
################################################################################
################################################################################
################################################################################
'''
month = input('type the first three letters of the interested month: ')
pattern = '*'+month+'.22m'


def disp_aux(pattern):
    path = '/home/isaac/Escritorio/mag_plot/aux'
    filenames = next(os.walk(path))[2]
    date_list = fnmatch.filter(filenames, pattern)
#date_list = date_list.sort()
    print(date_list)
disp_aux(pattern)
'''
################################################################################
################################################################################
################################################################################
i_date = input("write initial date in format \n yyyy-mm-dd HH:MM:SS  " )
f_date = input("write final date in format \n yyyy-mm-dd HH:MM:SS  " )    
################################################################################
################################################################################
################################################################################
def coe_df(i_date, f_date):
#concatenate all data into one DataFrame
#df = pd.read_csv('/home/c-isaac/Escritorio/mag_plot/coe07jan.22m', header=1, delim_whitespace=True, skip_blank_lines=True)

#'/run/user/1001/gvfs/sftp:host=10.0.0.187/home/ccastellanos/Escritorio/proyecto/coeneo'
    #path        = '/run/user/1001/gvfs/sftp:host=132.248.208.46,user=visitante/data/magnetic_data/coeneo/2022'
    path = '/home/isaac/geomstorm/datos/coeneo'
#name = input()
    file_names  = glob.glob(path+'/*.23m')

    dfs_c         = []

    for file_name in file_names:
        df = pd.read_csv(file_name, header=1, delim_whitespace=True, \
        skip_blank_lines=True)
        dfs_c.append(df)
    
    df_c = pd.concat(dfs_c, axis=0, ignore_index=True)

    dd = df_c.iloc[:,0].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    mm = df_c.iloc[:,1].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    yy = df_c.iloc[:,2].astype(str)

    df_c['dt_tmp']= yy+mm+dd

#print(df)
    df_c['date']  = pd.to_datetime(df_c['dt_tmp'], format='%Y-%m-%d')
#print(df['date'])
    df_c['hr']    = pd.to_timedelta(df_c.iloc[:,3], unit = 'h')
    df_c['min']   = pd.to_timedelta(df_c.iloc[:,4], unit = 'm')

    df_c['Date']  = datetimes = df_c['date'] + df_c['hr'] + df_c['min']

    dec = df_c.iloc[:,5]
    H   = df_c.iloc[:,6] 
    Z   = df_c.iloc[:,7]
    I   = df_c.iloc[:,8]
    F   = df_c.iloc[:,9]
#H = H.replace({-9999.9:np.nan})
    df_c = df_c.sort_values(by="Date")
#print(pd.isna(df)) 
    return (df_c)
################################################################################
################################################################################
################################################################################            
################################################################################
################################################################################
################################################################################
df = coe_df(i_date, f_date)  
#df = df.set_index(date)
################################################################################
#Detrend Time series
################################################################################
#df['H(nT)'] = signal.detrend(df['H(nT)'])
#df = df.reset_index(drop=True, inplace=True)

#idx_1 = pd.date_range(start = pd.Timestamp(i_date), end = pd.Timestamp(f_date), freq='T')
#print(df)    

################################################################################
################################################################################
################################################################################
def introduce_nan(dataframe, indice): #esta sub rutina introduce los datetime faltantes, considerando los periodos en que no se registraron los datos en vez de solo pegarlos
    
    dataframe = dataframe.set_index(dataframe['Date'])
    dataframe = dataframe.reindex(indice)
    dataframe = dataframe.drop(columns=['DD', 'MM', 'YYYY', 'HH', 'MM.1', 'dt_tmp', 'hr', 'min','date', 'Date'])

    return(dataframe)
################################################################################
################################################################################
################################################################################

df_wnan = introduce_nan(df, idx_1) #dataframe que considera el tiempo en que no se registraron datos

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
H    = df_wnan['H(nT)']
#imprimir PWS de componente H
################################################################################
#Bt = B_longtime(df_wnan)
#H    = Bt['H(nT)']
#plt.plot(df_wnan.index, Bt)
#plt.show()
#medianas cada hora
#print(H[jump])
#print(H[jump])
#print(df_wnan['H(nT)'][jump])
#print(df_wnan['H(nT)'][jump])
'''
path_dst = '/home/c-isaac/Escritorio/mag_plot/dst/'
file_name2 = path_dst+'ASY_'+i_date+'_'+f_date+'m_P.dat'
df_dst = pd.read_csv(file_name2, header=24, sep='\s+', skip_blank_lines=True)


sym = df_dst['SYM-H'] 

plt.plot(df_wnan.index, H, linewidth=1.0, label='experimental DH')
#plt.plot(df_wnan.index, sym, 'k', linewidth=1.0, label='SYM-H')
plt.xlim([inicioc, finalc])
plt.grid()
plt.legend()
plt.show()
'''


inicioc = df_wnan.index[0]
finalc  = df_wnan.index[-1]

mask = (df_c['Date'] > i_date) & (df_c['Date'] <= f_date)
df_c = df_c[mask]

Date = df_c['Date'][mask]
H            = H[mask]


for i in range(0,len(idx),step):
    mask = (df['index'] >= idx[i]) & (df['index'] <= idx[i+fhour])
    df_new = df[mask]
   # sym_H  = df_new['SYM-H'].apply(lambda x: '{0:0>4}'.format(x))
    date = str(idx[i])
    date = date[0:10]
    
    name_new = file_type+'_'+date+'.txt'
    new_path = '/home/isaac/geomstorm/datos/'+\
    file_type+'/daily/'
    df_new = df_new.drop(columns=['DOY'])
    df_new.to_csv(new_path+name_new, sep= '\t', index=False)     
    
    
    
          






