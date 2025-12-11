import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from gicdproc import gic_qd
from season_preprocess import window_27

idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

stat  = ['LAV', 'QRO', 'RMY','MZT']
dir_path = f'/home/isaac/datos/gics_obs/qdl/'
iwindows, fwindows = window_27(idate, fdate)




qd_lav = gic_qd(idate, fdate, dir_path, 'LAV')
qd_qro = gic_qd(idate, fdate, dir_path, 'QRO')
qd_rmy = gic_qd(idate, fdate, dir_path, 'RMY')
qd_mzt = gic_qd(idate, fdate, dir_path, 'MZT')


fig, axes = plt.subplots(4, 1, figsize=(18, 16))

def format_window_label(iw, fw):
    iw_year = iw[:4]
    iw_doy = iw[5:7]
    fw_year = fw[:4]
    fw_doy = fw[5:7]
    
    if iw_year == fw_year:
        return f"{iw_year}/{iw_doy}-{fw_doy}"
    else:
        return f"{iw_year}/{iw_doy}\n{fw_year}/{fw_doy}"

# Crear las etiquetas y posiciones
x_labels = [format_window_label(iw, fw) for iw, fw in zip(iwindows, fwindows)]
x_ticks_positions = [i * 1440 + 720 for i in range(len(iwindows))]

# Calcular límites
x_min = 0
x_max = len(iwindows) * 1440

stations_data = [
    (qd_lav, 'LAV'),
    (qd_qro, 'QRO'),
    (qd_rmy, 'RMY'),
    (qd_mzt, 'MZT')
]

for idx, (data, station) in enumerate(stations_data):
    axes[idx].plot(data.iloc[:,0], color='black', linewidth=2, label=f' QD model')
    axes[idx].plot(data.iloc[:,0] + data.iloc[:,1], color='red', alpha=0.7, linewidth=1, label=f'{station} ± Uncertainty')
    axes[idx].plot(data.iloc[:,0] - data.iloc[:,1], color='red', alpha=0.7, linewidth=1)
    axes[idx].set_ylabel(f'{station} - GIC QD Model [A]')
    axes[idx].set_xticks(x_ticks_positions)
    axes[idx].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[idx].set_xlim(x_min, x_max)  # Límites consistentes
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('/home/isaac/QD_gics.png', dpi=300, bbox_inches='tight')
plt.close()
#for i in range(len(stat)):
#for h in range(4):

def extract_amplitudes(data):
    amplitudes = []
    max_val  = []
    min_val = []
    for i in range(38):
        # Extract the window of data
        window_data = data[i*1440:(i+1)*1440]
        
        # Check if more than 50% of values are not NaN
        non_nan_count = np.count_nonzero(~np.isnan(window_data))
        
        if non_nan_count > 0.5 * len(window_data):  # More than 50% non-NaN
            tmp_max = np.nanmax(window_data)  # Use nanmax to ignore NaN values
            max_val.append(tmp_max)
            
            tmp_min = np.nanmin(window_data)  # Use nanmin to ignore NaN values
            min_val.append(tmp_min)
            
            amplitudes.append(tmp_max - tmp_min)
        else:
            # If less than 50% valid data, append NaN or skip
            max_val.append(np.nan)
            min_val.append(np.nan)
            amplitudes.append(np.nan)

    return amplitudes, max_val, min_val
qro_amplitudes, qro_max, qro_min = extract_amplitudes(qd_qro)
rmy_amplitudes, rmy_max, rmy_min = extract_amplitudes(qd_rmy)
mzt_amplitudes, mzt_max, mzt_min = extract_amplitudes(qd_mzt)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Panel 1: QRO Amplitudes
axes[0].plot(qro_amplitudes, color='red', linewidth=2, marker='o', markersize=4)
axes[0].set_title('QRO - Amplitudes')
axes[0].set_ylabel('Amplitude GIC [A]')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(len(qro_amplitudes)))

# Panel 2: RMY Amplitudes
axes[1].plot(rmy_amplitudes, color='black', linewidth=2, marker='s', markersize=4)
axes[1].set_title('RMY - Amplitudes')
axes[1].set_ylabel('Amplitude GIC [A]')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(len(rmy_amplitudes)))

# Panel 3: MZT Amplitudes
axes[2].plot(mzt_amplitudes, color='blue', linewidth=2, marker='^', markersize=4)
axes[2].set_title('MZT - Amplitudes')
axes[2].set_ylabel('Amplitude GIC[A]')
axes[2].set_xlabel('Window Index')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(range(len(mzt_amplitudes)))

plt.tight_layout()
plt.show()