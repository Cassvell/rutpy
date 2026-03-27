import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def tanh_func(x,a,b):
    
    return (a)* np.tanh(x) + b*x -10
    
def fit_tan(x_data, y_data):
    popt, pcov = curve_fit(tanh_func, x_data, y_data)
    a, b = popt
    print(f"Optimal parameters: a={a:.3f}, b={b:.3f}")
    print(f"Parameter errors: a_err={np.sqrt(pcov[0,0]):.3f}, b_err={np.sqrt(pcov[1,1]):.3f}")
    return tanh_func(x_data, -120, b), popt


#def get_chisq(model, data):
#    dY = data - model
#    return np.sum((dY / data)**2)

def LR(x, y):
    """Regresión lineal simple."""
    X = x.reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    print(f"Regresión lineal: pendiente={modelo.coef_[0]:.3f}, intercepto={modelo.intercept_:.3f}")
    return y_pred, modelo

dir_path = '/home/isaac/longitudinal_studio/'
df = pd.read_csv(f'{dir_path}ampere_j_net.csv', header=1, sep='\t')

I_net = df.iloc[:,1]
H_I = df.iloc[:,2]
rho = np.corrcoef(I_net, H_I)[0, 1]

xfit = np.linspace(np.min(I_net), np.max(I_net), len(I_net))
yfit, popt = fit_tan(xfit, H_I)
#chi_2 = get_chisq(yfit, H_I)

lmod, lr_model = LR(xfit, np.array(H_I))

plt.scatter(I_net, H_I, color='blue')
plt.plot(xfit, yfit, color='black')
#plt.plot(xfit, lmod*(-1), 'k--')

#plt.plot(xfit, y_fit1, 'r-')
#plt.plot(xfit, y_fit2, 'r--')

plt.xlabel(f'$I_{{net}}$ [M A]', fontsize=16)
plt.ylabel(f'$H_I$ [nT]', fontsize=16)
plt.xlim(-2.2, 2.2)
plt.xticks(fontsize=16)
plt.ylim(-150,150)
plt.yticks(fontsize=16)
plt.text(0.95, 0.95, rf'$\rho = {rho:.2f}$', transform=plt.gca().transAxes, fontsize=18, ha='right', va='top')

#plt.text(0.95, 0.85, rf'$\chi^2 = {chi_2:.2f}$', transform=plt.gca().transAxes,  fontsize=16, ha='right', va='top')
plt.tight_layout()
plt.savefig(f'{dir_path}/fig/jrvshi_2.png', dpi=300)
plt.show()
