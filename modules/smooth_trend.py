from scipy.interpolate import CubicSpline
from scipy import optimize
import sys
from scipy.interpolate import splrep, BSpline
from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt
def spl_fit(x,y,x_interp, w):
    # Eliminar puntos NaN para el ajuste del spline
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    w_clean = w[mask]
    # Verificar que haya suficientes puntos para la interpolación
    if len(x_clean) < 4:  # Mínimo de puntos para spline cúbico
        print(f"Advertencia: Solo {len(x_clean)} puntos válidos. Usando interpolación lineal.")
        return np.interp(x_interp, x_clean, y_clean)
    
    #try:
    f = 0.2

    s = (len(x_clean)/2) * np.var(y_clean) * f
    tck = splrep(x_clean, y_clean, k=3, w=w_clean, s=s)
    yfit = BSpline(*tck)(x_interp)
    return yfit
    
   # except Exception as e:
   #     print(f"Error en spline: {e}. Usando interpolación lineal.")
   #     return np.interp(x_interp, x_clean, y_clean)

    
    
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


def fourier_series_with_freqs(x, freqs):
    """
    Returns a symbolic fourier series with specified frequencies.
    
    :param x: Independent variable (tiempo en segundos)
    :param freqs: Array de frecuencias en Hz
    """
    # Separar frecuencia cero (término constante) de las no cero
    zero_mask = freqs == 0
    non_zero_freqs = freqs[~zero_mask]
    
    n = len(non_zero_freqs)
    
    # Crear todos los parámetros de una vez
    if n > 0:
        # Crear nombres para los coeficientes
        cos_names = [f'a_{i}' for i in range(1, n + 1)]
        sin_names = [f'b_{i}' for i in range(1, n + 1)]
        
        # Crear todos los parámetros
        all_params = parameters(','.join(['a0'] + cos_names + sin_names))
        
        # Separar los parámetros
        a0 = all_params[0]
        cos_coeffs = all_params[1:n+1]
        sin_coeffs = all_params[n+1:2*n+1]
    else:
        # Solo término constante
        a0 = parameters('a0')[0]
        cos_coeffs = []
        sin_coeffs = []
    
    # Comenzar con el término constante
    series = a0
    
    # Agregar términos de Fourier
    for i, (freq, ai, bi) in enumerate(zip(non_zero_freqs, cos_coeffs, sin_coeffs)):
        omega_t = 2 * np.pi * freq * x
        series += ai * cos(omega_t) + bi * sin(omega_t)
    
    return series



def fourier_fit(xdata, ydata, new_x, n_order):
    x, y = variables('x, y')
    model_dict = {y: fourier_series(x, f=2*np.pi, n=n_order)}
    
    fit = Fit(model_dict, x=xdata, y=ydata)
    fit_result = fit.execute()
    
    new_y = fit.model(x=new_x, **fit_result.params).y
    return new_y


def fourier_fit_with_freqs(tdata, ydata, new_t, freqs):
    """
    Fit data using Fourier series with specified frequencies.
    
    :param tdata: Tiempo en segundos
    :param ydata: Datos a ajustar
    :param new_t: Nuevos tiempos para predicción
    :param freqs: Array de frecuencias en Hz
    """
    t, y = variables('t, y')
 
    # Crear modelo con las frecuencias especificadas
    
    model_dict = {y: fourier_series_with_freqs(t, np.array(freqs))}
    
    # Realizar el ajuste
    fit = Fit(model_dict, t=tdata, y=ydata)
    fit_result = fit.execute()
    
    # Predecir
    new_y = fit.model(t=new_t, **fit_result.params).y
    return new_y