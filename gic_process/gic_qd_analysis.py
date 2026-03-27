import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.dates as mdates
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
    
    gic_data = data.iloc[:,0]
    baseline = data.iloc[:,2]
    
    for i in range(42):
        # Extract the window of data
        window_data = gic_data[i*1440:(i+1)*1440]
        window_baseline = baseline[i*1440:(i+1)*1440]
        # Check if more than 50% of values are not NaN
        non_nan_count = np.count_nonzero(~np.isnan(window_data))
        
        if non_nan_count > 0.5 * len(window_data):  # More than 50% non-NaN
            tmp_max = np.nanmax(window_data)  # Use nanmax to ignore NaN values
            max_val.append(tmp_max)
            
            tmp_min =  np.nanmax(window_baseline)  # Use nanmin to ignore NaN values
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


def season_qd_2(struct):
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
        
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))

    x_min = 0
    x_max = 23
    x = np.linspace(0,24,1440)
    equinox = [3,4,5,6,10,11,12,13,17,18,19,24,25,26,31,32,33,37,38,39,40]
    solsticio = [1,2,7,8,9,14,15,16,20,21,22,23,27,28,29,30,34,35,36,41,42]
    
    eq_prim = [3,4,5,6,17,18,19,31,32,33]
    eq_ot = [10,11,12,13,24,25,26,37,38,39,40]
    
    sol_inv = [1,2,14,15,16,27,28,29,30,41,42,43]
    sol_ver = [7,8,9,20,21,22,23,34,35,36]
    
    for idx, (station, data) in enumerate(struct.items()):
        data_st = data.iloc[:,0]
        
        axes[0, 0].set_title(f'Equinox seasons', fontsize=12, fontweight='bold')
    # Right subplot (solstice)
        axes[0, 1].set_title(f'Solstice seasons', fontsize=12, fontweight='bold')
        added_labels_eq = set()
        added_labels_sol = set()
        for j in range(1, 42):
            daily_model = data_st[j*1440:(j+1)*1440]
            
            # Check if current day index is in equinox or solstice
            if j in equinox:
                # Plot in left subplot (equinox)
                if j in eq_prim: 
                    color = 'green'
                    label = 'Spring'
                elif j in eq_ot:
                    
                    color = 'brown'
                    label = 'Fall'
                
                if label not in added_labels_eq:
                    axes[idx, 0].plot(x, daily_model, color=color, label=label, linewidth=1, alpha=0.7)
                    added_labels_eq.add(label)
                else:
                    axes[idx, 0].plot(x, daily_model, color=color, linewidth=1, alpha=0.7)
                
                
                
            elif j in solsticio:
                
                if j in sol_inv:
                    color = 'blue'
                    label='Winter'
                elif j in sol_ver:
                    color = 'darkorange'
                    label = 'Summer'
                if label not in added_labels_eq:
                    axes[idx, 1].plot(x, daily_model, color=color, label=label, linewidth=1, alpha=0.7)
                    added_labels_eq.add(label)
                else:
                    axes[idx, 1].plot(x, daily_model, color=color, linewidth=1, alpha=0.7)
    
    
    
        # Configure left subplot (equinox)
        axes[idx, 0].set_ylabel(f'{station} - GIC QD model [A]', fontsize=15)
        axes[idx, 0].set_xlim(0, x_max)
        axes[idx, 0].set_xlabel('UT [h]', fontsize=15)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx,0].legend(fontsize=15)
        axes[idx, 0].tick_params(axis='both', labelsize=14)
        # Configure right subplot (solstice)
        axes[idx, 1].set_ylabel(f'{station} - GIC QD model [A]', fontsize=15)
        axes[idx, 1].set_xlim(0, x_max)
        axes[idx, 1].set_xlabel('UT [h]', fontsize=15)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend(fontsize=15)
        axes[idx, 1].set_ylabel('')  
        axes[idx, 1].set_yticks([])  
        axes[idx, 1].tick_params(axis='both', labelsize=14)
        
        if idx == 3:  # Última fila
            axes[idx, 0].set_xlabel('Universal Time [h]', fontsize=15)
            axes[idx, 1].set_xlabel('Universal Time [h]', fontsize=15)
        else:
            # Ocultar xlabel en las filas superiores
            axes[idx, 0].set_xlabel('')
            axes[idx, 1].set_xlabel('')
            # Opcional: también ocultar los xticks para una apariencia más limpia
            axes[idx, 0].set_xticklabels([])
            axes[idx, 1].set_xticklabels([])

        
    plt.tight_layout()
    plt.savefig('/home/isaac/gics_rv/fig/QD_seasonal_stacked_e_s.png', dpi=300)
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
        
        median_amp = np.nanmedian(np.array(struct[estacion]['amplitudes']))
        print(f'{estacion} median amplitudes: {median_amp}')        
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

 
def baseline_study(data, med_windows, fit_type):

    x_min = 0
    x_max = len(data) * 1440 
    medwindows_doy = [int(str(d).split('-')[1]) for d in med_windows]

    decimal_doy = [i/365 for i in medwindows_doy]    
    baseline_points = []
    
    non_nan_ratio = np.sum(~np.isnan(data)) / len(data)
    
    if non_nan_ratio == 1:
        ini = 0
        fin = len(data)
        tot_data = fin-ini
    else:        
        is_nan = np.isnan(data)
        changes = np.diff(np.concatenate(([True], is_nan, [True])))
        indices = np.where(changes == True)
        ending = np.where(changes == 1)[0] - 1
        indices = np.array(indices).flatten()
        ini = indices[0]
        fin = indices[1]                     
        tot_data = fin-ini

    baseline_points = data[ini:fin]
    x_original = np.arange(ini, fin)
    x_data = (x_original*1440)
    new_xdata = np.linspace(ini*1440, fin*1440, fin*1440, dtype=int)
    
    #x_fit = np.linspace(ini, fin, tot_data*1440)
    f_cubic = interpolate.interp1d(x_data, decimal_doy[ini:fin], kind='cubic', fill_value="extrapolate")
    doy_fit = f_cubic(new_xdata)
    
    y_data = np.array(baseline_points)


    if fit_type == 'fourier':
        from scipy.signal import medfilt
        y = fourier_fit(np.array(decimal_doy[ini:fin]), y_data, doy_fit, n_order=6)
        y_fit = medfilt(y, kernel_size=1441)
        
    if fit_type == 'spl':
        f = 0.1
        w, desv = weights(y_data, 3, 3)
        s = (len(x_data)/2) * np.var(y_data) * f
        
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
    station_colors = {
    'LAV': 'blue',      # Reemplaza con el nombre real de tu estación
    'QRO': 'darkorange',       # Reemplaza con el nombre real de tu estación
    'RMY': 'green',     # Reemplaza con el nombre real de tu estación
    'MZT': 'purple'}
    
    years = {'2023': [0, 14], '2024': [14, 27], '2025': [27, 41]}
    for year_key in years.keys():
        # Crear figura con subplots para las 4 estaciones (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()  # Aplanar para facilitar el acceso

        # Procesar cada estación
        for idx, (station, data) in enumerate(stat_dir.items()):
            print(f"Procesando {station} para {year_key}")
            station_color = station_colors[station]

            ini = 0
            fin = 42
            
            # Preparar datos base
            base_data = []
            for w in range(len(iwindows)):
                pos = w * 1440
                tmp = data.iloc[pos, 2]
                base_data.append(tmp)
            
            # Obtener índices para el año actual
            year_value = years[year_key]
            ini_index = year_value[0]
            fin_index = year_value[1]
            yearly_data = base_data[ini_index:fin_index]
            
            # Calcular ratio de datos no NaN
            non_nan_ratio = np.sum(~np.isnan(yearly_data)) / len(yearly_data)
            
            # Obtener DOY para el año actual
            doy = med_windows_doy[ini_index:fin_index]
            xaxis = np.arange(0, len(doy))
            xaxis_sc = xaxis*1440
  
            # Graficar si hay suficientes datos
            if non_nan_ratio > 0.5:
                x, y, new_x, new_y = baseline_study(yearly_data, doy, 'fourier')
                x2, y2, new_x2, new_y2 = baseline_study(yearly_data, doy, 'spl')
                
                # Graficar datos originales y ajustes
                axes[idx].plot(x, y, color=station_color ,marker='o', markersize=8, linestyle='', label='Original data')
                axes[idx].plot(new_x, new_y, color=station_color, linestyle='-', linewidth=1, label='Fourier fit')
                axes[idx].plot(new_x2, new_y2, color=station_color, linestyle='-.', linewidth=2, label='Spline fit')
                axes[idx].set_xlim(xaxis_sc[0], xaxis_sc[-1])
            num_points = len(doy)
            tick_indices = [0, num_points//4, num_points//2, 3*num_points//4, num_points-1]
            tick_positions = [xaxis_sc[idx] for idx in tick_indices if idx < len(xaxis_sc)]
            tick_labels = [doy[idx] for idx in tick_indices if idx < len(doy)]
            
            axes[idx].set_xticks(tick_positions)
            axes[idx].set_xticklabels(tick_labels, rotation=0, ha='right', fontsize=10)
            
            axes[idx].legend(fontsize=10)
            axes[idx].set_ylabel(f'{station} [A]', fontsize=13)
            axes[idx].set_xlabel('Day of the year', fontsize=13)
            axes[idx].grid(True, alpha=0.3)
            
            # Opcional: Configurar ticks si es necesario
            # axes[idx].set_xlim(0, 42)
            # axes[idx].set_xticks(range(0, 43, 7))
            # axes[idx].set_xticklabels(range(0, 43, 7))
        
        # Ajustar el layout general de la figura
        plt.suptitle(f'Seasonal Variation - year {year_key}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Guardar la figura para el año actual
        plt.savefig(f'/home/isaac/gics_rv/fig/seasonal_var_{year_key}.png', dpi=300, bbox_inches='tight')

        plt.close() 

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
#plot_season = season_qd_2(stat_dir)
#plot_amp = season_amp(amp_dir)

plot_baseline = baseline_plot(stat_dir)

sys.exit('end')
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
    ax.set_title(f'GIC detector: {station}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Universal Time [h]', fontsize=16)
    ax.set_ylabel(r'$\sigma_{stack}$ [A]', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=16)
    
    ticks = df_station.index[::4]
    labels = [t[0:2] for t in ticks] 
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=16)
    max_val = df_station['std_30min'].max()
    max_idx = df_station['std_30min'].idxmax()

# Ajustar el layout para evitar superposiciones
plt.tight_layout()
plt.savefig('/home/isaac/gics_rv/fig/erroresSQ.png', dpi=300)
plt.show()
#CALCULAR LA VARIACION APILADA CADA MEDIA HORA DE VARIACIONES.
#QUIZA GENERAR UNA GRAFICA DE LAS VARIACIONES APILADAS POR CADA ESTACION. UN SUBPLOT DE 2,2
#
#
