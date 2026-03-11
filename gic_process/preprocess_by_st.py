import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data
from modules.calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt
import os

idate = sys.argv[1]
fdate = sys.argv[2]

fyear = int(fdate[0:4])
fmonth = int(fdate[4:6])
fday = int(fdate[6:8])

finaldate= datetime(fyear, fmonth,fday)

finaldate= datetime(fyear, fmonth,fday)
nextday = finaldate+timedelta(days=1)
nextday = str(nextday)[0:10]
idx1 = pd.date_range(start = pd.Timestamp(idate+ ' 12:01:00'), \
                          end = pd.Timestamp(nextday + ' 12:00:00'), freq='min')

ndays = calculate_days_difference(idate, fdate)
tot_data = (ndays+1)*1440


stat = 'MZT'
path = f'/home/isaac/datos/gics_obs/'

idx1 = pd.date_range(start = pd.Timestamp(idate+ ' 00:00:00'), \
                          end = pd.Timestamp(fdate + ' 23:59:00'), freq='min')

ndays = calculate_days_difference(idate, fdate)
tot_data = (ndays+1)*1440

file = []
dict_gic = {'MZT': []}
gic_dic = {'MZT': []}
'''
for i in stat:
    print(f'station:{i}')
    data = df_gic_pp(idate, fdate, path, i)
    #print(data.index)
    #plt.plot(data['gic'], label = f'{i}')
    
#plt.show()
sys.exit('pruebas para leer pp')
'''

print(f'station:{stat}')
gic_st, T1TW, T2TW = process_station_data(idate, fdate, path, stat, idx1, tot_data)



dict_gic = {'gic' : gic_st, 'T1' : T1TW, 'T2' : T2TW}

df_st = pd.DataFrame(dict_gic)   
    
df_st['gic'] = np.where((df_st['gic'] >= 400) | (df_st['gic'] <= -400), np.nan, df_st['gic'])
#plt.plot(df_st['gic']) 
    
new_idate = df_st.index[0] + pd.Timedelta(hours=12, minutes=00)
new_fdate = df_st.index[-1] - pd.Timedelta(hours=12, minutes=00)

df_shifted = df_st[new_idate:new_fdate] 


df_shifted.index = df_shifted.index - pd.Timedelta(hours=7)
fyear = int(idate[0:4])
fmonth = int(idate[4:6])
fday = int(idate[6:8])
finaldate= datetime(fyear, fmonth,fday)
nextidate = finaldate+timedelta(days=1)
nextidate = str(nextidate)[0:10]
date_range = pd.date_range(start=nextidate + ' 00:00:00', end=nextday+ ' 23:59:00', freq='min')
df_shifted = df_shifted[~df_shifted.index.duplicated(keep='first')]
df_shifted = df_shifted.reindex(date_range)
plt.plot(df_shifted['gic'], color='r')
#plt.xlim(idx1[0], idx1[-1])
#plt.tight_layout()
plt.show()


header = " ".join(f"{key:>10}" for key in dict_gic.keys())

header = f"{'Datetime':>7}{'gic':>20}{'T1':>13}{'T2':>15}"
    #print(df_shifted.index)
    
if df_shifted.isna().any().any():
    # Fill numeric columns with -999.999 and object columns with a string placeholder
    numeric_cols = df_shifted.select_dtypes(include=[np.number]).columns
    object_cols = df_shifted.select_dtypes(include=['object']).columns

#print

for j in range(ndays):
    start_idx = j * 1440
    end_idx = (j + 1) * 1440
    
    if end_idx > len(df_shifted):
        print(f"Skipping {j}, index out of range")
        continue

    # Slice daily data
    daily_data = df_shifted.iloc[start_idx:end_idx].copy()
    daily_data = daily_data.reset_index()     
    daily_data = daily_data.rename(columns={'index':'Datetime'})
    date = daily_data['Datetime'].iloc[0]
    
    tmp_year = date.year
    tmp_month = date.month
    tmp_day = date.day
    # Convert datetime to timestamp
    #daily_data['Datetime'] = daily_data['Datetime'].apply(
    #    lambda x: 999.9 if pd.isna(x) else x.timestamp()

    
    # Fill NaN values
    daily_data_filled = daily_data.fillna(999.9)
    
    output_dir = f'/home/isaac/datos/gics_obs/{tmp_year}/{stat}/daily/'
    filename = f"{stat}_{date.strftime('%Y-%m-%d')}.pp.csv"
    filepath = os.path.join(output_dir, filename)
    print(daily_data_filled)
    sys.exit('end')
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        daily_data_filled.to_csv(filepath, index=False)
        print(f'Archivo creado: {filepath} ({len(daily_data_filled)} registros)')
    except Exception as e:
        print(f"Error al guardar {filepath}: {e}")
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################


    #print(f"Created empty files for station {stat} with NaN values") 

    #print(df_shifted)
    
    #dict_gic[i] = {'gic' : gic_st, 'T1' : T1TW, 'T2' : T2TW}



#fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))  # 3 rows, 1 column#


#ax1.plot(gic_dic['LAV']['gic'].index, gic_dic['LAV']['gic'], label='LAV', color='blue', alpha=0.7)
#ax1.set_xlim(gic_dic['LAV']['gic'].index[0],gic_dic['LAV']['gic'].index[-1])
#ax1.legend()
#ax1.set_ylabel('GIC')

#ax2.plot(gic_dic['QRO']['gic'].index, gic_dic['QRO']['gic'], label='QRO', color='orange', alpha=0.7)
#ax2.set_xlim(gic_dic['QRO']['gic'].index[0], gic_dic['QRO']['gic'].index[-1])
#ax2.legend()
#ax2.set_ylabel('GIC')

#ax3.plot(gic_dic['RMY']['gic'].index, gic_dic['RMY']['gic'], label='RMY', color='green', alpha=0.7)
#ax3.set_xlim(gic_dic['RMY']['gic'].index[0], gic_dic['RMY']['gic'].index[-1])
#ax3.legend()
#ax3.set_ylabel('GIC')

#ax4.plot(gic_dic['MZT']['gic'].index, gic_dic['MZT']['gic'], label='MZT', color='red', alpha=0.7)
#ax4.set_xlim(gic_dic['MZT']['gic'].index[0], gic_dic['MZT']['gic'].index[-1])
#ax4.legend()
#ax4.set_ylabel('GIC')

#plt.tight_layout()

#plt.show()

             



        #plt.plot(gic_res, label=f'{i} GIC no Diurnal Base', alpha=0.7)
