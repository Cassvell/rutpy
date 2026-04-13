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
n_cols = 2
n_rows = (2) 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()  

for idx, st in enumerate(stations):
    df = pd.read_csv(f'{dir_path}{st}_thresholds.minmax.csv', header=0, sep=',')
    media = df.iloc[:,1]
    FWHM = df.iloc[:,2]
    acc95 = df.iloc[:,3]
    #df.set_index(medwindows)
    print(f'{st}')
    print(rf'MEDIA: {np.nanmedian(media):.2f}, FWHM*2: {np.nanmedian(FWHM):.2f}, $\sigma$ FWHM*2: {np.nanstd(FWHM):.2f}, acc95: {np.nanmedian(acc95):.2f}, $\sigma$ acc95: {np.nanstd(acc95):.2f}')
    
    median_fwhm = np.nanmedian(FWHM)
    std_fwhm = np.nanstd(FWHM)
    median_acc95 = np.nanmedian(acc95)
    std_acc95 = np.nanstd(acc95)

    #print(media)
    # Eliminar valores NaN
    mask_FWHM = ~np.isnan(FWHM)
    mask_acc95 = ~np.isnan(acc95)
    
    x_points = np.arange(len(FWHM))
    
    coef = linreg(mask_FWHM)
    
    #y = coef[0]*x_points+coef[1]
    
    xfixed1 = np.full(len(media), 0.3)
    xfixed2 = np.full(len(media), 0.6)

    axes[idx].plot(xfixed1, FWHM, 'ko', markersize=12, alpha=0.2, label=r'$FWHM$')    
    axes[idx].plot(xfixed2, acc95, 'ro', markersize=12, alpha=0.2, label= r'$95\%$ cdf')    
    
    axes[idx].errorbar(0.3, median_fwhm, yerr=std_fwhm, 
                   fmt='ko', markersize=20, capsize=10, capthick=2,
                   elinewidth=2, markeredgewidth=2, alpha=1.0)

    axes[idx].errorbar(0.6, median_acc95, yerr=std_acc95, 
                   fmt='ro', markersize=20, capsize=10, capthick=2,
                   elinewidth=2, markeredgewidth=2, alpha=1.0)    
    
    # Configuraciones del gráfico
    axes[idx].set_xlim(0, 1)
    axes[idx].tick_params(axis='both', which='major', labelsize=15)
    axes[idx].set_ylabel(f'{st} Thresholds [A]', fontsize=15)
    axes[idx].grid(True, alpha=0.3)
    
    if idx >= len(stations) - n_cols:
    #    axes[idx].set_xticks(xticks_positions)
    #    axes[idx].set_xticklabels(xticks_labels, ha='left', fontsize=15)
    #    axes[idx].set_xlabel('Windows [days]', fontsize=15)
        axes[idx].legend(fontsize=15)
    #else:
        
    axes[idx].set_xticklabels([])

# Ocultar subplots vacíos
for idx in range(len(stations), len(axes)):
    axes[idx].set_visible(False)

#plt.suptitle('Thresholds dispersion', fontsize=24)
plt.tight_layout()
plt.savefig(f'{dir_path}fig/whitenoise.minmax.png', dpi=300)
plt.close()

