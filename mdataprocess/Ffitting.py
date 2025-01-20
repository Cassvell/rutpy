from symfit import parameters, variables, sin, cos, Fit
import numpy as np
from scipy.interpolate import splrep, splev

#import matplotlib.pyplot as plt


def fourier_series(x, f, n=0):

    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    phase_angles = parameters(','.join(['phi{}'.format(i) for i in range(1, n + 1)]))

    # Construct the series with phase angles
    series = a0 + sum(ai * cos(i * f * x + phi) 
                     for i, (ai, phi) in enumerate(zip(cos_a, phase_angles), start=1))
    return series

def fit_data(data):
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, f=w, n=8)}
    print(model_dict)

    ndata = len(data)
    xdata = np.linspace(0, ndata - 1, ndata)
    ydata = data

    # Define a Fit object for this model and data
    fit = Fit(model_dict, x=xdata, y=ydata)
    fit_result = fit.execute()
    print(fit_result)
    fit_data = fit.model(x=xdata, **fit_result.params).y
    
    return fit_data