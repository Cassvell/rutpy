import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys

path = f'/home/isaac/MEGA/posgrado/doctorado/semestre5/'

df = pd.read_csv(f'{path}gics_tabla.csv', sep=',', header = 1)
df = df.replace('n/r', np.nan)  
df = df.replace('n/a', np.nan)    



df = df.where(df.iloc[:,2] >-75 ,np.nan)

#print(df.iloc[:,2:6])  


#sys.exit('end of test')





lav = pd.to_numeric(df.iloc[:,3], errors='coerce')
qro = pd.to_numeric(df.iloc[:,4], errors='coerce')
rmy = pd.to_numeric(df.iloc[:,5], errors='coerce')
mzt = pd.to_numeric(df.iloc[:,6], errors='coerce')






def distribution_threshold_plot(lav, name):
    lav = lav[~np.isnan(lav)]
    lav = np.unique(lav)    
    median_value = np.median(lav)
    print(f"Median value: {median_value:.4f}")
    frequencies, bin_edges = np.histogram(lav, bins=60, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def lorentzian(x, amplitude, x0, gamma):
        """Lorentzian (Cauchy) distribution function"""
        return amplitude * (gamma**2 / ((x - x0)**2 + gamma**2))

    # Generate histogram data for fitting
    counts, bin_edges = np.histogram(lav, bins=45, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial parameter guesses
    initial_guess = [
        np.max(counts),        # amplitude (peak height)
        np.median(lav),        # x0 (location parameter)
        np.std(lav) / 2        # gamma (scale parameter)
    ]

    # Fit the Lorentzian curve
    try:
        popt, pcov = curve_fit(lorentzian, bin_centers, counts, p0=initial_guess, maxfev=5000)
        amplitude, x0, gamma = popt
        
        print(f"Fitted Lorentzian parameters:")
        print(f"Amplitude: {amplitude:.4f}")
        print(f"Location (x0): {x0:.4f}")
        print(f"Scale (gamma): {gamma:.4f}")
        print(f"FWHM: {2*gamma:.4f}")
        print(f"Median: {median_value:.4f}")
        
    except RuntimeError:
        print("Fit failed - trying alternative initial parameters")
        initial_guess_alt = [np.max(counts), np.mean(lav), np.std(lav)]
        popt, pcov = curve_fit(lorentzian, bin_centers, counts, p0=initial_guess_alt, maxfev=5000)
        amplitude, x0, gamma = popt

    # Create fitted curve
    x_fit = np.linspace(np.min(lav), np.max(lav), 1000)
    y_fit = lorentzian(x_fit, amplitude, x0, gamma)

    # Plot the results
    plt.figure(figsize=(12, 7))
    plt.hist(lav, bins=45, density=True, alpha=0.7, label='Data', color='lightblue', edgecolor='black')
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Cauchy fit\nx₀={x0:.2f}, γ={gamma:.2f}')

    # Add vertical line for median
    plt.axvline(x=median_value, color='green', linestyle='--', linewidth=2, 
                label=f'Median: {median_value:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Cauchy Distribution Fit for {name.upper()} st')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{path}/{name}_dist.png')
    plt.show()

    
lav_dist = distribution_threshold_plot(lav, 'lav')
mzt_dist = distribution_threshold_plot(mzt, 'mzt')
qro_dist = distribution_threshold_plot(qro, 'qro')
rmy_dist = distribution_threshold_plot(rmy, 'rmy')