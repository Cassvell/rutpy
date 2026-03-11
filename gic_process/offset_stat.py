import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                         
from datetime import timedelta

fname = '/home/isaac/rutpy/gic_process/gics_offsetchange.csv'
df = pd.read_csv(fname, sep=',', header=0)
t0 = df.iloc[:,0]
hour = []
for t in t0:
    if pd.notna(t):  
        horas = int(t * 24)

        minutos_decimal = (t * 24) - horas
        minutos = int(round(minutos_decimal * 60))

        if minutos == 60:
            horas += 1
            minutos = 0
        td = timedelta(hours=horas, minutes=minutos)
        hour.append(td)
        #hour.append(td)     
    else:
        hour.append(pd.NaT) 
 
dt = df.iloc[:,2]
dt_sec = dt.astype(str) + ':00'
dt_formated = pd.to_timedelta(dt_sec) 
        
horas_t0 = []
horas_dt = []

# Para t0
for td in hour:
    if pd.notna(td):
        horas_enteras = td.total_seconds() / 3600
        horas_t0.append(horas_enteras)

# Para dt_formated
for td in dt_formated:
    if pd.notna(td):
        horas_enteras = td.total_seconds() / 3600
        horas_dt.append(horas_enteras)

# Crear dos histogramas separados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histograma para t0
ax1.hist(horas_t0, bins=24, edgecolor='black', alpha=0.7, color='skyblue')
ax1.axvline(x=np.mean(horas_t0), color='red', linestyle='--', label=f'Media: {np.mean(horas_t0):.2f}h')
ax1.axvline(x=np.median(horas_t0), color='green', linestyle='--', label=f'Mediana: {np.median(horas_t0):.2f}h')
ax1.set_xlabel(r'cambio de offset $t_0$ [h]')
ax1.set_ylabel('Frecuencia')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histograma para dt
ax2.hist(horas_dt, bins=24, edgecolor='black', alpha=0.7, color='lightcoral')
ax2.axvline(x=np.mean(horas_dt), color='red', linestyle='--', label=f'Media: {np.mean(horas_dt):.2f}h')
ax2.axvline(x=np.median(horas_dt), color='green', linestyle='--', label=f'Mediana: {np.median(horas_dt):.2f}h')
ax2.set_xlabel('duracion de cambio de offset [h]')
ax2.set_ylabel('Frecuencia')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
