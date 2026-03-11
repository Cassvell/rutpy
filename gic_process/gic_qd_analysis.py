import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy import interpolate

import sys
from datetime import datetime, timedelta
from gicdproc import gic_qd
from modules.window_27 import window_27
from modules.smooth_trend import splrep
from scipy.interpolate import splrep, BSpline
from sklearn.metrics import r2_score, mean_squared_error
from symfit import parameters, variables, sin, cos, Fit
idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

def fourier_series(x, f, n):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    cos_a = parameters(','.join(['a{}'.format(i) for i in range(1, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                 for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series



def fourier_fit(xdata, ydata, new_x, n_order):
    x, y = variables('x, y')
    model_dict = {y: fourier_series(x, f=2*np.pi, n=n_order)}
    
    fit = Fit(model_dict, x=xdata, y=ydata)
    fit_result = fit.execute()
    
    new_y = fit.model(x=new_x, **fit_result.params).y
    return new_y

def extract_amplitudes(data):
    amplitudes = []
    max_val  = []
    min_val = []
    for i in range(42):
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

def season_qd(struct):
    iwindows, medwindows, fwindows, nwindows= window_27(idate, fdate, 'date')
    
    grid_positions = []
    for w in range(len(iwindows)):
        pos = w * 1440   # Centro de cada ventana
        grid_positions.append(pos)

    xticks_positions = []
    xticks_labels = []

    for w in range(0, nwindows, 3):  # Cada 3 ventanas
        # Posición en el medio de la ventana
        pos = w * 1440 
        
        # Formato yyyy1mm1dd1-yyyy2mm2dd2
        date = medwindows[w].strftime('%Y%m%d')

        label = f'{date}'
        
        xticks_positions.append(pos)
        xticks_labels.append(label)    
        
    fig, axes = plt.subplots(4, 1, figsize=(18, 16))

    x_min = 0
    x_max = nwindows * 1440

    for idx, (station, data) in enumerate(struct.items()):
        print(np.max(data.iloc[:,0]))
        
        # Gráficas continuas (cada 1440 puntos por ventana)
        axes[idx].plot(data.iloc[:,0], color='black', linewidth=2, label=f'QD model')
        axes[idx].plot(data.iloc[:,0] + data.iloc[:,1], color='red', alpha=0.7, linewidth=1, label=f'{station} ± Uncertainty')
        axes[idx].plot(data.iloc[:,0] - data.iloc[:,1], color='red', alpha=0.7, linewidth=1)
        
        axes[idx].set_ylabel(f'{station} - GIC QD Model [A]', fontsize=12)
        axes[idx].set_xlim(x_min, x_max)
        
        axes[idx].set_xticks(xticks_positions)
        axes[idx].set_xticklabels(xticks_labels, ha='center', fontsize=12)
        
        # Configurar el grid vertical en las posiciones de las ventanas
        axes[idx].grid(True, alpha=0.3, which='major')
        axes[idx].set_xticks(grid_positions, minor=False)
        axes[idx].xaxis.grid(True, which='major', linestyle='-', alpha=0.5)    
        axes[idx].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()    


def season_amp(struct):    
    
    iwindows, medwindows, fwindows, nwindows= window_27(idate, fdate, 'date')
    
    grid_positions = []
    for w in range(len(iwindows)):
        pos = w * 1440   # Centro de cada ventana
        grid_positions.append(pos)

    xticks_positions = []
    xticks_labels = []
    x_original = np.arange(0, 42)
    x_data = (x_original*1440+720)
    for w in range(0, nwindows, 3):  # Cada 3 ventanas
        # Posición en el medio de la ventana
        pos = w * 1440 
        
        # Formato yyyy1mm1dd1-yyyy2mm2dd2
        date = medwindows[w].strftime('%Y%m%d')

        label = f'{date}'
        
        xticks_positions.append(pos)
        xticks_labels.append(label)        
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Lista de estaciones en orden
    estaciones = ['LAV', 'QRO', 'RMY', 'MZT']

    # Colores y marcadores para cada estación
    estilos = {
        'LAV': {'color': 'red', 'marker': 'o'},
        'QRO': {'color': 'black', 'marker': 's'},
        'RMY': {'color': 'blue', 'marker': '^'},
        'MZT': {'color': 'green', 'marker': 'D'}
    }

    for idx, estacion in enumerate(estaciones):
        axes[idx].plot(x_data, struct[estacion]['amplitudes'], 
                    color=estilos[estacion]['color'], 
                    linewidth=2, 
                    marker=estilos[estacion]['marker'], 
                    markersize=4)
        axes[idx].set_title(f'{estacion}', fontsize=16)
        axes[idx].set_xlim(x_data[0], x_data[-1])
        axes[idx].set_ylabel('Amplitude GIC [A]', fontsize=12)        
        axes[idx].set_xticks(xticks_positions)
        axes[idx].set_xticklabels(xticks_labels, ha='left', fontsize=12)
        
        # Configurar el grid vertical en las posiciones de las ventanas
        axes[idx].grid(True, alpha=0.3, which='major')
        axes[idx].set_xticks(grid_positions, minor=False)
        axes[idx].xaxis.grid(True, which='major', linestyle='-', alpha=0.5)    
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()    

 
def baseline_study(data, med_windows, fit_type, ini, fin):

    x_min = 0
    x_max = fin * 1440
    x_original = np.arange(ini, fin)
    
    x_data = (x_original*1440+720)
    
    new_xdata = np.arange(ini*1440+720, (fin*1440)+720)     
    medwindows_doy = [int(str(d).split('-')[1]) for d in med_windows]

    decimal_doy = [i/365 for i in medwindows_doy]    
    baseline_points = []
    
    for w in x_original:
        baseline_value = data.iloc[w * 1440, 2]
        baseline_points.append(baseline_value)

    tot_data = fin-ini
    x_fit = np.linspace(ini, fin, tot_data*1440)
    f_cubic = interpolate.interp1d(x_original, decimal_doy[ini:fin], kind='cubic', fill_value="extrapolate")
    doy_fit = f_cubic(x_fit)
    
    # Puntos originales (suponiendo que son 42 puntos igualmente espaciados)
    
    y_data = np.array(baseline_points)

    if fit_type == 'fourier':
        y_fit = fourier_fit(np.array(decimal_doy[ini:fin]), y_data, doy_fit, n_order=5)
    if fit_type == 'spl':
        f = 0.1
        w, desv = weights(y_data, 3, 3)
        s = (len(x_data)/2) * np.var(y_data) * f
        print(s)
        tck = splrep(x_data, y_data, w=w, s=s, k=3)
        y_fit = BSpline(*tck)(new_xdata)
        
        #metricas = metricas_ajuste(baseline_points, y_fit[x_data], len(y_data[x_data]))
        #for nombre, valor in metricas.items():
        #    print(f"{nombre}: {valor:.4f}")

    return x_data, baseline_points, new_xdata, y_fit


def baseline_plot(data):
    iwindows_doy, med_windows_doy, fwindows_doy, nwindows= window_27(idate, fdate, 'doy')
    iwindows, med_windows, fwindows, nwindows= window_27(idate, fdate, 'date')
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    grid_positions = []
    for w in range(len(iwindows)):
        pos = w * 1440   # Centro de cada ventana
        grid_positions.append(pos)

    xticks_positions = []
    xticks_labels = []

    for w in range(0, nwindows, 6):  # Cada 3 ventanas
        # Posición en el medio de la ventana
        pos = w * 1440 
        
        # Formato yyyy1mm1dd1-yyyy2mm2dd2
        med_date = med_windows[w].strftime('%Y%m%d')
        
        label = f'{med_date}'
        
        xticks_positions.append(pos)
        xticks_labels.append(label)        
    
    for idx, (station, data) in enumerate(stat_dir.items()):
        print(station)

        ini = 0
        fin = 42   
            
        if station == 'RMY':
            fin = 27
            
        if station == 'MZT':
            ini = 5
            fin = 34
        
        x, y, new_x, new_y = baseline_study(data, med_windows_doy,'fourier', ini, fin)
        x2,y2, new_x2, new_y2 = baseline_study(data, med_windows_doy,'spl', ini, fin)
        
        axes[idx].plot(x, y, 'ko')
        
        axes[idx].plot(new_x, new_y, color='blue', linewidth=1) 
        axes[idx].plot(new_x2, new_y2, color='red', linewidth=1)  
        axes[idx].set_ylabel(f'{station} - GIC QD Model [A]', fontsize=12)
        axes[idx].set_xlim(0, 42*1440)
        axes[idx].set_xticks(xticks_positions)
        axes[idx].set_xticklabels(xticks_labels, ha='left', fontsize=12)
        
        # Configurar el grid vertical en las posiciones de las ventanas
        axes[idx].grid(True, alpha=0.3, which='major')
        axes[idx].set_xticks(grid_positions, minor=False)
        axes[idx].xaxis.grid(True, which='major', linestyle='-', alpha=0.5)    
        axes[idx].grid(True, alpha=0.3)        
    plt.show()

def weights(y, ventana=3, umbral=3):
    """
    Asigna pesos usando una función de Tukey (biweight)
    para ser más robusto ante outliers.
    """
    n = len(y)
    residuos = np.zeros(n)
    
    # Calcular residuos respecto a promedio local
    for i in range(n):
        inicio = max(0, i - ventana)
        fin = min(n, i + ventana + 1)
        
        if fin - inicio > 2:
            indices = list(range(inicio, fin))
            indices.remove(i)
            prom_local = np.median(y[indices])  # Usar mediana, más robusta
            residuos[i] = abs(y[i] - prom_local)
    
    # Escala robusta (MAD = Median Absolute Deviation)
    escala = np.median(residuos[residuos > 0]) * 1.4826
    
    if escala == 0:
        escala = 1
    
    # Normalizar residuos
    u = residuos / (escala * umbral)
    
    # Función de peso de Tukey (biweight)
    pesos = np.zeros_like(u)
    mascara = np.abs(u) < 1
    pesos[mascara] = (1 - u[mascara]**2)**2
    
    # Asegurar que todos tengan al menos un peso pequeño
    pesos = np.maximum(pesos, 0.01)
    
    return pesos, residuos

def metricas_ajuste(y_real, y_pred, n_params):
    n = len(y_real)
    
    # R² (Coeficiente de determinación)
    r2 = r2_score(y_real, y_pred)
    
    # R² ajustado
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_real - y_pred))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    
    # AIC (Akaike Information Criterion)
    rss = np.sum((y_real - y_pred)**2)
    aic = n * np.log(rss/n) + 2 * n_params
    
    # BIC (Bayesian Information Criterion)
    bic = n * np.log(rss/n) + n_params * np.log(n)
    
    return {
        'R²': r2,
        'R² Ajustado': r2_ajustado,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'AIC': aic,
        'BIC': bic
    }


stat  = ['LAV', 'QRO', 'RMY', 'MZT']
dir_path = f'/home/isaac/datos/gics_obs/qdl/'

#PRIMERA COLUMNA: MODELO DE VARIACION DIURNA
#SEGUNDA COLUMNA: VARIACION HORA A HORA
#TERCERA COLUMNA: LINEA BASE DE LA VENTANA DE TIEMPO
stat_dir = {}
amp_dir = {}
baselines = {}

for st in stat:
    #print(f'{st}')
    window_data = gic_qd(idate, fdate, dir_path, st, 'gic')
    window_data = window_data.replace(999.9, np.nan)
    stat_dir[st] = window_data
    qd_amplitudes, st_max, st_min = extract_amplitudes(window_data)
    amp_dir[st] = {
                    'amplitudes': qd_amplitudes,
                    'max': st_max,
                    'min': st_min}

    
#plot_season = season_qd(stat_dir)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes_flat = axes.flatten()
for idx, (station, data) in enumerate(stat_dir.items()):
    print(station)
    variation_data = data.iloc[:,1]
    ndata = len(variation_data)
    nmods = int(ndata/1440)
    
    
    daily_columns = []
    for i in range(nmods):
        daily_sample = variation_data[i*1440:(i+1)*1440]
        #hh_sample = []
        hh_sample = daily_sample[::30].tolist()
        daily_columns.append(hh_sample)
        
    df_station = pd.DataFrame(daily_columns).T
    df_station.index = [f'{h:02d}:{m:02d}' for h in range(24) for m in (0, 30)]

    df_station['std_30min'] = np.nanstd(df_station, axis=1)

    
    ax = axes_flat[idx]
    
    # Graficar la desviación estándar
    ax.plot(df_station.index, df_station['std_30min'], 
            marker='o', markersize=3, linewidth=1.5, color='blue', alpha=0.7)
    
    ax.set_xlim(df_station.index[0], df_station.index[-1])
    ax.set_title(f'GIC detector: {station}', fontsize=14, fontweight='bold')
    ax.set_xlabel('UT [h]', fontsize=14)
    ax.set_ylabel(r'$\sigma_{stack}$ [A]', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotar etiquetas del eje x para mejor legibilidad
    ax.tick_params(axis='x', rotation=45)
    
    # Mostrar solo algunas etiquetas para no saturar
    ax.set_xticks(df_station.index[::4])  # Mostrar cada 4ta etiqueta (cada 2 horas)
    
    # Añadir valor máximo como anotación
    max_val = df_station['std_30min'].max()
    max_idx = df_station['std_30min'].idxmax()

# Ajustar el layout para evitar superposiciones
plt.tight_layout()
plt.close()
#CALCULAR LA VARIACION APILADA CADA MEDIA HORA DE VARIACIONES.
#QUIZA GENERAR UNA GRAFICA DE LAS VARIACIONES APILADAS POR CADA ESTACION. UN SUBPLOT DE 2,2
plot_amp = season_amp(amp_dir)
#plot_baseline = baseline_plot(stat_dir)
