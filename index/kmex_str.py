import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

date = sys.argv[1]# "formato(yyyymm)"
ext_H = '.delta_H.early'
ext_K = '.k_index.early'
stat = 'teo_'
path = '/home/isaac/datos/monthly_index/'

df_K = pd.read_csv(f'{path}{stat}{date}{ext_K}', header=21, sep='\s+')
df_H = pd.read_csv(f'{path}{stat}{date}{ext_H}', header=21, sep='\s+')



df_K = df_K.drop("|", axis=1)
k = df_K.iloc[:,2:9]
k[k > 90] = np.nan  # or pd.NA (for nullable integers)
        
k_max = k.max(axis=1)
k_sum = df_K.iloc[:,10]
Date = df_K.iloc[:,0]

dH = df_H.iloc[:,2:26]
dH[dH == 999999.0] = np.nan  # or pd.NA (for nullable integers)

dH_min = dH.min(axis=1)
dH_date = df_H.iloc[:,0]

result = pd.DataFrame({
    'Date_K': Date,
    'K_max': k_max/10,
    'K_sum': k_sum/10,
    'Date_dH': dH_date,
    'dH_min': dH_min
})


print(result)