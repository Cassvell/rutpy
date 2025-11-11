import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from gicdproc import  process_station_data, df_gic_pp
from calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt
from gic_threshold import threshold
from gic_diurnalbase import gic_diurnalbase
from corr_offset import corr_offset
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


stat = ['MZT', 'LAV', 'QRO', 'RMY']
#stat = ['MZT', 'QRO', 'RMY', 'MZT']
#st = ['QRO', 'QRO', 'RMY', 'MZT']
path = f'/home/isaac/datos/gics_obs/'

idx1 = pd.date_range(start = pd.Timestamp(idate+ ' 00:00:00'), \
                          end = pd.Timestamp(fdate + ' 23:59:00'), freq='min')

ndays = calculate_days_difference(idate, fdate)
tot_data = (ndays+1)*1440

file = []
dict_gic = {'MZT': [], 'LAV' : [],  'QRO' : [], 'RMY': []}
gic_dic = {'MZT': [], 'LAV' : [],  'QRO' : [],  'RMY': []}
'''
for i in stat:
    print(f'station:{i}')
    data = df_gic_pp(idate, fdate, path, i)
    #print(data.index)
    #plt.plot(data['gic'], label = f'{i}')
    
#plt.show()
sys.exit('pruebas para leer pp')
'''
for i in stat:
    print(f'station:{i}')
    gic_st, T1TW, T2TW = process_station_data(idate, fdate, path, i, idx1, tot_data)

    if not gic_st.isnull().all():

        dict_gic[i] = {'gic' : gic_st, 'T1' : T1TW, 'T2' : T2TW}
        
        df_st = pd.DataFrame(dict_gic[i])   
        #plt.plot()     
        if i == 'LAV':
            #plt.plot(gic_st, label=f'{i} GIC ', color='black', alpha=0.7)
            gic_resample = gic_st.resample('30Min').median().fillna(method='ffill')
            
            threshold = threshold(gic_resample, idate, fdate, stat, '2s')   
            stddev = np.nanstd(gic_resample)         
            

            #for j in range(ndays):
                #start_idx = j * 1440
                #end_idx = (j + 1) * 1440
                #gic_corr = corr_offset(df_st['gic'][start_idx:end_idx], threshold, 60, stddev)
                #df_st['gic'][start_idx:end_idx] = gic_corr 
                #second_filter = corr_offset(gic_st[start_idx:end_idx], threshold, 60, stddev)
                #gic_st[start_idx:end_idx] = second_filter 
        df_st['gic'] = np.where((df_st['gic'] >= 400) | (df_st['gic'] <= -400), np.nan, df_st['gic'])
        #plt.plot(df_st['gic'], label=f'{i} GIC ', color='red', alpha=0.7)
        #plt.show()

            
        new_index = df_st.index + pd.Timedelta(hours=12, minutes=00)
        df_shifted = pd.DataFrame(data=df_st.values, index=new_index, columns=df_st.columns)
        plt.plot(df_shifted['gic'], label=f'{i} GIC ', color='blue', alpha=1)            
        plt.show()
        header = " ".join(f"{key:>10}" for key in dict_gic[i].keys())

        header = f"{'Datetime':>7}{'gic':>20}{'T1':>13}{'T2':>15}"
            #print(df_shifted.index)
        
        if df_shifted.isna().any().any():
            # Fill numeric columns with -999.999 and object columns with a string placeholder
            numeric_cols = df_shifted.select_dtypes(include=[np.number]).columns
            object_cols = df_shifted.select_dtypes(include=['object']).columns
        
        #if len(numeric_cols) > 0:
            #df_shifted[numeric_cols] = df_shifted[numeric_cols].fillna(999.9)       
    
        gic_dic[i] = {'gic' : df_shifted['gic'], 'T1' : df_shifted['T1'], 'T2' :  df_shifted['T2']}
        #sys.exit('end')
        for j in range(ndays):
            start_idx = j * 1440
            end_idx = (j + 1) * 1440
            
            if end_idx > len(df_shifted):
                print(f"Skipping {j}, index out of range")
                continue

            # Slice daily data
            daily_data = df_shifted[start_idx:end_idx].reset_index()     
            date = daily_data.iloc[:,0]
            tmp_year = date.iloc[0].year
            tmp_month = date.iloc[0].month
            tmp_day = date.iloc[0].day
            daily_data = daily_data.rename(columns={'index':'Datetime'})

            # Convert datetime to timestamp
            #daily_data['Datetime'] = daily_data['Datetime'].apply(
            #    lambda x: 999.9 if pd.isna(x) else x.timestamp()
        
            
            # Fill NaN values
            daily_data_filled = daily_data.fillna(999.9)
            
            output_dir = f'/home/isaac/datos/gics_obs/{tmp_year}/{i.upper()}/daily/'
            filename = f"{i}_{date.iloc[0].strftime('%Y-%m-%d')}.pp.csv"
            filepath = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Export using to_csv with tab separator
            daily_data_filled.to_csv(
                filepath,
                sep='\t',
                header=True,
                index=False,
                float_format='%12.7f'  # Consistent formatting
            )        
    
        if i == 'MZT':
            
            df_shifted.index = df_shifted.index - pd.Timedelta(hours=7)
            fyear = int(idate[0:4])
            fmonth = int(idate[4:6])
            fday = int(idate[6:8])
            finaldate= datetime(fyear, fmonth,fday)
            nextidate = finaldate+timedelta(days=1)
            nextidate = str(nextidate)[0:10]
            date_range = pd.date_range(start=nextidate + ' 00:00:00', end=nextday+ ' 23:59:00', freq='1min')
            df_shifted = df_shifted[~df_shifted.index.duplicated(keep='first')]
            df_shifted = df_shifted.reindex(date_range)
        
        

                
                #for _, row in daily_data_filled.iterrows():
                #    datetime_str = row['Datetime'].strftime('%Y-%m-%d %H:%M:%S')
                #    gic_str = f"{row['gic']:12.7f}"
                #    t1_str = f"{row['T1']:12.7f}"
                #    t2_str = f"{row['T2']:12.7f}"
                #    f.write(f"{datetime_str}   {gic_str}   {t1_str}   {t2_str}\n")

    else:
        
        fyear = int(idate[0:4])
        fmonth = int(idate[4:6])
        fday = int(idate[6:8])
        finaldate= datetime(fyear, fmonth,fday)
        nextidate = finaldate+timedelta(days=1)
        nextidate = str(nextidate)[0:10]
        date_range = pd.date_range(start=nextidate + ' 00:00:00', end=nextday+ ' 23:59:00', freq='1min')    

        # Create DataFrame with DateTime index and NaN values for data columns
        empty_data = pd.DataFrame({
            'gic':  np.full(1440*(ndays+1), 999.9),
            'T1': np.full(1440*(ndays+1), 999.9),
            'T2': np.full(1440*(ndays+1), 999.9)
        })
        empty_data = empty_data.set_index(date_range)
        
        # Apply the same time shift
        new_index = empty_data.index + pd.Timedelta(hours=24, minutes=00)
        df_shifted = pd.DataFrame(data=empty_data.values, index=new_index, columns=empty_data.columns)        

        header = f"{'Datetime':>20}{'gic':>15}{'T1':>15}{'T2':>15}"
        
        for j in range(ndays):
            start_idx = j * 1440
            end_idx = (j + 1) * 1440
            
            if end_idx > len(df_shifted):
                print(f"Skipping {j}, index out of range")
                continue

            # Slice daily data
            daily_data = df_shifted[start_idx:end_idx].reset_index()     
            date = daily_data.iloc[:,0]
            tmp_year = date.iloc[0].year
            tmp_month = date.iloc[0].month
            tmp_day = date.iloc[0].day
            daily_data = daily_data.rename(columns={'index':'Datetime'})

            # Convert datetime to timestamp
            #daily_data['Datetime'] = daily_data['Datetime'].apply(
            #    lambda x: 999.9 if pd.isna(x) else x.timestamp()
            #)
            
            # Fill NaN values
            daily_data_filled = daily_data.fillna(999.9)
            
            output_dir = f'/home/isaac/datos/gics_obs/{tmp_year}/{i.upper()}/daily/'
            filename = f"{i}_{date.iloc[0].strftime('%Y-%m-%d')}.pp.csv"
            filepath = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Export using to_csv with tab separator
            daily_data_filled.to_csv(
                filepath,
                sep='\t',
                header=True,
                index=False,
                float_format='%12.7f'  # Consistent formatting
            )
        
        print(f"Created empty files for station {i} with NaN values") 

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
