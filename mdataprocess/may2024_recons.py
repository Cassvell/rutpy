import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sys 
from datetime import datetime, timedelta
from aux_time_DF import index_gen, convert_date
idate = sys.argv[1]
fdate = sys.argv[2]

dir_path = '/home/isaac/datos/regmex/itu/'


filenames_out = []

idx = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(str(fdate) + ' 23:59:00'), freq='min')
 
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(str(fdate) + ' 23:59:00'), freq='D')
dates = []
#df = 
dfs_c = []
for i in idx_daily:
        date_name = str(i)[0:10]
        dates.append(date_name)
        date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
        filenames = f'itu_{date_name_newf}M.dat'
        #print(f'{dir_path}experiment_itu/{filenames}')
        df_frag = pd.read_csv(f'{dir_path}experiment_itu/{filenames}', header = 0, sep = '\\s+')
        df_frag = df_frag.replace(999.9, np.nan)    
        dfs_c.append(df_frag)
        #
        # print(date_name_newf)
may_data = pd.concat(dfs_c, axis=0, ignore_index=True)   
may_data = may_data.set_index(idx)
df = pd.read_csv(f'{dir_path}interp_disturbance.csv', header=0, sep=',')

plt.plot(df['ITUX'], label='X')
plt.plot(df['ITUY'], label='Y')
plt.legend()
plt.show()
sys.exit('end')
ndf = len(df)
ndays = int(ndf/1440)

ini_df = '2024-02-01 00:00:00'
end_df = datetime.strptime(str(20240201), '%Y%m%d') + timedelta(days=ndays-1, hours=23, minutes=59)

idx2 = pd.date_range(start = pd.Timestamp(ini_df), \
                    end = pd.Timestamp(end_df), freq='min')
df = df.set_index(idx2)
may_ini = '2024-05-01 00:00:00'
may_end = '2024-05-31 23:59:00'
may_window = df[may_ini:may_end]

D_base_rad = may_data['Dbase']  # Already in radians, no conversion needed
X_base = may_data['Hbase'] * np.cos(D_base_rad)
Y_base = may_data['Hbase'] * np.sin(D_base_rad)
Z_base = may_data['Zbase']



# SQ station
D_sq_rad = may_data['DSQ'] 
X_sq = may_data['HSQ'] * np.cos(D_sq_rad)
Y_sq = may_data['HSQ'] * np.sin(D_sq_rad)
Z_sq = may_data['ZSQ']

#I = np.tan(Z/H)




dD = np.arctan2(may_window['ITUY'], may_window['ITUX'])
x_raw = may_window['ITUX'] + X_base + X_sq
y_raw = may_window['ITUY'] + Y_base + Y_sq
z_raw = may_window['ITUZ'] + Z_base + Z_sq

H_raw = np.sqrt(x_raw**2 + y_raw**2)
D_raw = np.arctan2(y_raw, x_raw)


fig, axes = plt.subplots(3, 3, figsize=(15, 10))

# Row 1: Differences and SQ components (originalmente fila 2)
axes[0, 0].plot(may_window['ITUX'], label=r'$\Delta X$', color='blue')
axes[0, 0].plot(X_sq, label=r'$SQ X$', color='red')
axes[0, 0].set_title(r'$\Delta X$ and SQ X')
axes[0, 0].set_ylabel('nT')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(may_window['ITUY'], label=r'$\Delta Y$', color='orange')
axes[0, 1].plot(Y_sq, label=r'$SQ Y$', color='red')
axes[0, 1].set_title(r'$\Delta Y$ and SQ Y')
axes[0, 1].set_ylabel('nT')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[0, 2].plot(may_window['ITUZ'], label=r'$\Delta Z$', color='green')
axes[0, 2].plot(Z_sq, label=r'$SQ Z$', color='red')
axes[0, 2].set_title(r'$\Delta Z$ and SQ Z')
axes[0, 2].set_ylabel('nT')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Row 2: Raw components (originalmente fila 1)
axes[1, 0].plot(x_raw)
axes[1, 0].set_title('x_raw')
axes[1, 0].set_ylabel('nT')
axes[1, 0].grid(True)

axes[1, 1].plot(y_raw)
axes[1, 1].set_title('y_raw')
axes[1, 1].set_ylabel('nT')
axes[1, 1].grid(True)

axes[1, 2].plot(z_raw)
axes[1, 2].set_title('z_raw')
axes[1, 2].set_ylabel('nT')
axes[1, 2].grid(True)

# Row 3: Derived parameters (sin cambios)
axes[2, 0].plot(H_raw, label=r'$H_{raw} recons$', color='blue')
axes[2, 0].plot(may_data['H']+ may_data['HSQ']+may_data['Hbase'], label=r'$H_{raw}$', color='red')
axes[2, 0].set_title(r'$H_{raw}$')
axes[2, 0].set_ylabel('nT')
axes[2, 0].legend()
axes[2, 0].grid(True)

# Convert to degrees for better interpretation
dD_deg = np.degrees(dD)
DSQ_deg = may_data['DSQ'] / 60  # Convert minutes to degrees

axes[2, 1].plot(dD_deg, label=r'$\Delta D$', color='orange')
axes[2, 1].plot(DSQ_deg, label=r'$SQ dec$', color='red')
axes[2, 1].set_title(r'Declination')
axes[2, 1].set_ylabel('Degrees')
axes[2, 1].legend()
axes[2, 1].grid(True)

D_raw_deg = np.degrees(D_raw)
axes[2, 2].plot(D_raw_deg, label=r'$D_{raw}$', color='green')
axes[2, 2].plot(may_data['D']+may_data['Dbase']+may_data['DSQ'], label=r'$D_{raw2}$', color='red')
axes[2, 2].set_title(r'$D_{raw}$')
axes[2, 2].set_ylabel('Degrees')
axes[2, 2].legend()
axes[2, 2].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()