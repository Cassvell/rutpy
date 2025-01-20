import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from magnetic_datstruct import get_dataframe
from aux_time_DF import index_gen, convert_date

from datetime import datetime, date, timedelta
from magdata_processing import get_qd_dd, max_IQR

idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

dirpath = '/home/isaac/test_isaac/'
df = pd.read_excel(dirpath+'qdl.ods', engine='odf', sheet_name=0)
#print(len(df.))

idx = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='T')
idx_daily = pd.date_range(start = pd.Timestamp(str(idate)), \
                        end = pd.Timestamp(str(fdate)), freq='D')

#filenames = []
path = '/home/isaac/MEGAsync/datos/jicamarca/huan/'
st = 'huan'

filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)


data = get_dataframe(filenames, path, idx, dates)

arr = np.full(((len(df.columns)-1), 10, 1440), 999.9)
column = df.columns[1:-1]


for i, col in enumerate(column):
    qd = [0] * 10
    qdl = [[0] * 1440 for _ in range(10)]
    
    for j in range(10):
        qd[j] = df[col].iloc[j]  
       
        if qd[j] is not pd.NaT:
            qd[j] = str(qd[j])[0:10]
            
            qdl[j] = data.get(qd[j], None)
            
            if qdl[j] is not None:
                qdl[j] = qdl[j].reset_index(drop=True)
                
                if len(qdl[j]) >= 480:  
                    qd_2h = qdl[j][300:480]
                    
                    qdl[j] = qdl[j] - np.nanmedian(qd_2h)
                    
                    arr[i, j, :] = qdl[j]  
                    #plt.plot(arr[i, j, :], label=f"Column {col} Day {j+1}")

#plt.legend()
#plt.show()
print(arr)
np.save('/home/isaac/test_isaac/X_huan_2023_2024.npy', arr)