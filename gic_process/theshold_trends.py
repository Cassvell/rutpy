import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from symfit import parameters, variables, sin, cos, Fit
import seaborn as sns
from modules.window_27 import window_27
import sys
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
    # Convertir a arrays numpy y asegurar que son 1D
    xdata = np.array(xdata).flatten()
    ydata = np.array(ydata).flatten()
    new_x = np.array(new_x).flatten()
    
    x, y = variables('x, y')
    model_dict = {y: fourier_series(x, f=2*np.pi/len(xdata), n=n_order)}  # Frecuencia ajustada
    
    try:
        fit = Fit(model_dict, x=xdata, y=ydata)
        fit_result = fit.execute()
        
        # Evaluar el modelo
        new_y = fit.model(x=new_x, **fit_result.params).y
        return new_y
    except Exception as e:
        print(f"Error en fourier_fit: {e}")
        # Devolver ceros en caso de error
        return np.zeros_like(new_x)


def linreg(data):
    n = len(data)
    y = np.array(data)
    x = np.arange(0, n)
    X = np.column_stack([np.ones(n), x])
    coef = np.linalg.inv(X.T @ X) @ X.T @ y
    intercepto = coef[0]
    pendiente = coef[1]
    y_pred = X @ coef
    residuos = y - y_pred
    
    ss_res = np.sum(residuos ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    median_res = np.median(residuos)
    return residuos, median_res
idate = sys.argv[1]
fdate = sys.argv[2]

dir_path = '/home/isaac/gics_rv/'

stations = ['LAV', 'QRO', 'RMY', 'MZT']
# Crear figura con subplots
n_plots = len(stations)
n_cols = 1
n_rows = (n_plots + n_cols - 1) 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

# Crear puntos para el ajuste suave
x_data_smooth = np.linspace(0, 40, 200)  # Menos puntos para mayor suavidad

iwindows, medwindows, fwindows, nwindows= window_27(idate, fdate, 'date')

grid_positions = []
for w in range(len(iwindows)):
    pos = w * 1440   # Centro de cada ventana
    grid_positions.append(pos)

xticks_positions = []
xticks_labels = []
x_original = np.arange(0, 42)
x_data = (x_original*1440+720)
for w in range(0, nwindows, 4):  # Cada 3 ventanas
    # Posición en el medio de la ventana
    pos = w * 1440 
    
    # Formato yyyy1mm1dd1-yyyy2mm2dd2
    date = medwindows[w].strftime('%Y%m%d')

    label = f'{date}'
    
    xticks_positions.append(pos)
    xticks_labels.append(label)          

for idx, st in enumerate(stations):
    df = pd.read_csv(f'{dir_path}{st}_thresholds.minmax.csv', header=0, sep=',')
    media = df.iloc[:,1]
    FWHM = df.iloc[:,2]
    acc95 = df.iloc[:,3]
    #df.set_index(medwindows)
    print(f'{st}')
    print(rf'MEDIA: {np.nanmedian(media):.2f}, FWHM*2: {np.nanmedian(FWHM):.2f}, $\sigma$ FWHM*2: {np.nanstd(FWHM):.2f}, acc95: {np.nanmedian(acc95):.2f}, $\sigma$ acc95: {np.nanstd(acc95):.2f}')
    
    
    #print(media)
    # Eliminar valores NaN
    mask_FWHM = ~np.isnan(FWHM)
    mask_acc95 = ~np.isnan(acc95)
    
    x_points = np.arange(len(FWHM))
    
    coef = linreg(mask_FWHM)
    
    #y = coef[0]*x_points+coef[1]
    
    
    axes[idx].plot(x_data, media+FWHM, 'ko', markersize=8, label=r'$2 \cdot FWHM + \mu $')    
    axes[idx].plot(x_data, acc95, 'ro', markersize=8, label= r'$95\%$ cdf'    )
    # Configuraciones del gráfico
    #axes[idx].set_xlim(-1, len(FWHM))
    axes[idx].tick_params(axis='both', which='major', labelsize=15)
    axes[idx].set_ylabel(f'{st} Thresholds', fontsize=15)
    axes[idx].grid(True, alpha=0.3)
    
    if idx >= len(stations) - n_cols:
        axes[idx].set_xticks(xticks_positions)
        axes[idx].set_xticklabels(xticks_labels, ha='left', fontsize=15)
        axes[idx].set_xlabel('Windows [days]', fontsize=15)
        axes[idx].legend(fontsize=15)
    else:
        
        axes[idx].set_xticklabels([])

# Ocultar subplots vacíos
for idx in range(len(stations), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Thresholds by 27 days period', fontsize=24)
plt.tight_layout()
plt.savefig(f'{dir_path}fig/whitenoise.minmax.png', dpi=300)
plt.close()

