import numpy as np
#from lmoments3 import distr
import lmoments3 as lm
#import kneed as kn
#from scipy.stats import genpareto, kstest #anderson
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
import sys
import os
def gpd_cdf(x, k, sigma, mu):
    """
    Generalized Pareto Distribution CDF
    """
    x = np.array(x)
    z = (x - mu) / sigma
    
    if k == 0:
        return 1 - np.exp(-z)
    else:
        mask = (1 + k*z) > 0
        result = np.zeros_like(x)
        result[mask] = 1 - (1 + k*z[mask])**(-1/k)
        return result

def GPD_params(x):
    moments = lm.lmom_ratios(x, nmom=3)
    #print(f'L-moments: {moments}')
    
    t_3 = moments[2]/moments[1]
    
    k = (3*t_3 - 1) / (1 + t_3)
    s = moments[1]*(1 - k)*(2 - k)
    u = moments[0] - (s/(1 - k))
    
 
     
    return k, s, u 
     
        
def anderson_darling_r(data, k, sigma, mu):
    """
    Calculate Anderson-Darling statistic for GPD goodness-of-fit
    """
    # Transform to uniform using GPD CDF
    Z = gpd_cdf(data, k, sigma, mu)
    Z = np.clip(Z, 1e-10, 1-1e-10)  # Avoid boundary issues
    
    # Sort the transformed values
    Z_sorted = np.sort(Z)
    n = len(Z_sorted)
    
    # Calculate Anderson-Darling statistic
    total_sum = 0
    for i in range(1, n + 1):
        term1 = np.log(Z_sorted[i-1])
        term2 = np.log(1 - Z_sorted[n-i])
        total_sum += (2*i - 1) * (term1 + term2)
    
    A2 = -n - (1/n) * total_sum
    return A2


def gpd_ad_p_value_approximate(a2, n):
    """
    Compute p-value from GPD Anderson-Darling statistic
    """
    # Adjust for sample size
    a2_star = a2 * (1.0 + 0.2 / np.sqrt(n))
    
    # Different approximations for different A² ranges
    if a2_star < 0.2:
        p = 1.0 - np.exp(-13.436 + 101.14*a2_star - 223.73*a2_star**2)
    elif a2_star < 0.6:
        p = 1.0 - np.exp(-8.318 + 42.796*a2_star - 59.938*a2_star**2)
    elif a2_star < 3.0:
        p = np.exp(0.9177 - 4.279*a2_star - 1.38*a2_star**2)
    elif a2_star < 10.0:
        p = np.exp(1.2937 - 5.709*a2_star + 0.0186*a2_star**2)
    else:
        # For very large A² values
        p = np.exp(2.0 - 0.5*a2_star)
    
    return np.clip(p, 1e-100, 1.0)


def final_params(sorted_picks_norp, cdf,bound):
    
    idx = np.searchsorted(cdf, bound)
    
    # Initialize lists to store parameters
    AD_results = []
    p_values = []  # New list for p-values
    params_k = []
    params_s = []
    params_u = []
    start_indices = []
    sample_sizes = []  # Store sample sizes for reference

    for i in range(len(sorted_picks_norp[0:idx])):
        # Get subset of data from current index to bound
        data_subset = sorted_picks_norp[i:idx]
        n_subset = len(data_subset)
        
        if n_subset >= 3:  # Need at least 3 points for L-moments
            tmp_k, tmp_s, tmp_u = GPD_params(data_subset)
            
            # Calculate Anderson-Darling statistic
            a_tmp = anderson_darling_r(data_subset, tmp_k, tmp_s, tmp_u)
            
            # Calculate p-value from A² statistic
            p_tmp = gpd_ad_p_value_approximate(a_tmp, n_subset)
            
            # Store all results
            params_k.append(tmp_k)
            params_s.append(tmp_s)
            params_u.append(tmp_u)
            start_indices.append(i)
            AD_results.append(a_tmp)
            p_values.append(p_tmp)
            sample_sizes.append(n_subset)
            
            # Print results for this iteration
            #print(f"Start index {i}: k={tmp_k:.4f}, s={tmp_s:.4f}, u={tmp_u:.4f}, "
            #    f"n={n_subset}, A²={a_tmp:.4f}, p-value={p_tmp:.6f}")
                
        else:
            print(f"Skipping start index {i}: insufficient data points ({n_subset})")

    # Convert to numpy arrays for easier analysis
    params_k = np.array(params_k)
    params_s = np.array(params_s)
    params_u = np.array(params_u)
    start_indices = np.array(start_indices)
    AD_results = np.array(AD_results)
    p_values = np.array(p_values)
    sample_sizes = np.array(sample_sizes)
    
#    print(AD_results)
    u_candidates = []
    k_candidates = []
    s_candidates = []
    pval_candidates = []
    A_candidates = []
    for i in range(len(AD_results)):
       # if AD_results[i] < 10:
        if p_values[i] < 1 and p_values[i] > 0.7:
           # print(f'A²={AD_results[i]}, p-values={p_values[i]}, U= {params_u[i]}')
            u_candidates.append(params_u[i])
            pval_candidates.append(p_values[i])   
            A_candidates.append(AD_results[i])     
            k_candidates.append(params_k[i])
            s_candidates.append(params_s[i])
        
    if len(u_candidates) > 1:
        u_idx = np.argmax(np.array(u_candidates))
        u_max = np.max(np.array(u_candidates))
        definitive_params = {'u' : u_max, 'k' : k_candidates[u_idx], 's' : s_candidates[u_idx], \
            'A2' : A_candidates[u_idx], 'p' : pval_candidates[u_idx]}
    else:
        
        u_max = np.array(u_candidates)
        definitive_params = {'u' : u_max, 'k' : k_candidates[0], 's' : s_candidates[0], \
            'A2' : A_candidates[0], 'p' : pval_candidates[0]}

    return definitive_params


def gaussian(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def gaussian_fit(xdata, ydata):
    # Estimación inicial mejorada
    H0 = np.min(ydata)
    A0 = np.max(ydata) - np.min(ydata)
    
    # Encontrar el pico de manera más robusta
    max_idx = np.argmax(ydata)
    x0_est = xdata[max_idx]
    
    # Estimar sigma usando FWHM (Full Width at Half Maximum)
    half_max = (np.max(ydata) + np.min(ydata)) / 2
    indices_above_half = np.where(ydata >= half_max)[0]
    
    if len(indices_above_half) >= 2:
        # Ancho a mitad de altura
        fwhm = xdata[indices_above_half[-1]] - xdata[indices_above_half[0]]
        sigma0 = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Relación FWHM-sigma
    else:
        # Fallback: usar rango/4
        sigma0 = (xdata[-1] - xdata[0]) / 4
    
    # Asegurar sigma positivo en estimación inicial
    sigma0 = max(sigma0, (xdata[1] - xdata[0]))  # Mínimo el ancho de bin
    
    initial_guess = [H0, A0, x0_est, sigma0]
    
    # Límites más ajustados
    bounds_lower = [0, 0, xdata[0], sigma0/10]
    bounds_upper = [np.max(ydata), np.max(ydata)*2, xdata[-1], (xdata[-1]-xdata[0])]
    
    try:
        parameters, _ = curve_fit(gaussian, xdata, ydata, p0=initial_guess,
                                  bounds=(bounds_lower, bounds_upper))
    except:
        # Si falla, intentar sin bounds pero con sigma positivo forzado
        parameters, _ = curve_fit(gaussian, xdata, ydata, p0=initial_guess)
        parameters[3] = abs(parameters[3])  # Forzar sigma positivo
    
    H_fit, A_fit, x0_fit, sigma_fit = parameters
    fit_y = gaussian(xdata, H_fit, A_fit, x0_fit, sigma_fit)
    
    return fit_y, [H_fit, A_fit, x0_fit, sigma_fit]


def half_gaussian_fit(xdata, ydata):
    """
    Ajuste para distribución medio-gaussiana (half-normal)
    Asume que los datos son valores absolutos (x >= 0)
    """
    # Estimación inicial
    H0 = np.min(ydata)
    A0 = np.max(ydata) - np.min(ydata)
    # Para half-gaussian, el pico está en x=0
    # Estimamos sigma a partir del decaimiento
    half_max = np.max(ydata) / 2
    idx_half = np.where(ydata <= half_max)[0]
    if len(idx_half) > 0:
        sigma0 = xdata[idx_half[0]] / np.sqrt(2*np.log(2))
    else:
        sigma0 = (xdata[-1] - xdata[0]) / 4
    
    initial_guess = [H0, A0, sigma0]
    
    try:
        parameters, _ = curve_fit(half_gaussian, xdata, ydata, p0=initial_guess)
        return parameters
    except:
        print("Error en ajuste half-gaussian, usando valores iniciales")
        return initial_guess

def half_gaussian(x, H, A, sigma):
    """
    Distribución medio-gaussiana (half-normal)
    Para x >= 0, con media en 0
    """
    return H + A * np.exp(-(x)**2 / (2 * sigma**2))

def folded_gaussian(x, H, A, mu, sigma):
    """
    Distribución normal plegada (folded normal)
    Para |x| con media mu y sigma
    """
    return H + A * (1/np.sqrt(2*np.pi*sigma**2)) * (
        np.exp(-(x - mu)**2/(2*sigma**2)) + 
        np.exp(-(x + mu)**2/(2*sigma**2))
    )

def half_normal_cdf(x, sigma):
    """
    CDF de la distribución half-normal
    F(x) = erf(x/(σ√2)) para x ≥ 0
    """
    return erf(x / (sigma * np.sqrt(2)))

def half_normal_cdf_with_offset(x, sigma, offset):
    """
    CDF de half-normal con offset (por si la CDF no empieza en 0)
    """
    return offset + (1 - offset) * erf(x / (sigma * np.sqrt(2)))

def fit_half_normal_cdf(xdata, ydata):
    """
    Ajusta una CDF half-normal a los datos
    """
    # Estimación inicial de sigma usando el percentil 68% (equivalente a 1 sigma en half-normal)
    # En half-normal, el percentil 68% corresponde a ~1.6σ aproximadamente
    idx_68 = np.where(ydata >= 0.68)[0]
    if len(idx_68) > 0:
        x_68 = xdata[idx_68[0]]
        sigma0 = x_68 / np.sqrt(2)  # Aproximación inicial
    else:
        sigma0 = np.percentile(xdata, 95) / 2  # Fallback
    
    # Intentar ajuste con y sin offset
    try:
        # Primero intentar con offset
        params, _ = curve_fit(half_normal_cdf_with_offset, xdata, ydata, 
                             p0=[sigma0, 0], 
                             bounds=([1e-10, -0.1], [np.inf, 0.1]))
        sigma_fit, offset_fit = params
        fit_y = half_normal_cdf_with_offset(xdata, sigma_fit, offset_fit)
        return fit_y, sigma_fit, offset_fit
    except:
        # Si falla, intentar sin offset
        try:
            params, _ = curve_fit(half_normal_cdf, xdata, ydata, p0=[sigma0])
            sigma_fit = params[0]
            fit_y = half_normal_cdf(xdata, sigma_fit)
            return fit_y, sigma_fit, 0
        except:
            # Si todo falla, usar estimación inicial
            print("Warning: Ajuste de CDF falló, usando estimación inicial")
            sigma_fit = sigma0
            fit_y = half_normal_cdf(xdata, sigma_fit)
            return fit_y, sigma_fit, 0



def nbin_compute(data, ndata):
    norm_dist = pd.Series(data)
    q1 = norm_dist.quantile(0.25)
    q3 = norm_dist.quantile(0.75)
    iqr = q3-q1
    
    bin_width = (2*iqr) / ndata**(1/3)
    bin_count = int(np.ceil((norm_dist.max() - norm_dist.min()) / bin_width))
    return(bin_count)

def threshold(picks, i_date, f_date, st, method):
    picks_np_array = np.array(picks)
    picks_clean = picks_np_array[~np.isnan(picks_np_array)]
    sorted_picks = np.sort(np.abs(picks_clean))

    sorted_picks_norp = np.unique(sorted_picks)
    nbins = int(len(sorted_picks) / 3)

    stddev_res = (np.array(sorted_picks_norp).flatten())

    frequencies, bin_edges = np.histogram(stddev_res, bins=nbins*2, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = np.arange(1, len(sorted_picks_norp)+1) / len(sorted_picks_norp)

    #k, s, u = GPD_params(sorted_picks_norp)
    
    



   # optimal_mu, optimal_p, result = find_optimal_mu_constrained(sorted_picks_norp, k, s, u, n_simulations=500)
    bound = 95
    idx = np.percentile(sorted_picks, bound)
    if method == '2s':
        idx = np.percentile(sorted_picks, bound)

        threshold = idx
    elif method == '3s':
        bound =   99
        idx = np.percentile(sorted_picks, bound)

        threshold = idx
        
    elif method == 'GPD':
        results = []
        
        for i in range(len(sorted_picks_norp[0:idx])):
            data_subset = sorted_picks_norp[i:idx]
            n_subset = len(data_subset)
            
            if n_subset >= 3:
                k, s, u = GPD_params(data_subset)
                a2 = anderson_darling_r(data_subset, k, s, u)
                p_val = gpd_ad_p_value_approximate(a2, n_subset)
                
                results.append({
                    'start_index': i,
                    'k': k, 's': s, 'u': u,
                    'A2': a2, 'p_value': p_val,
                    'n_points': n_subset
                })
        
        if not results:
            raise ValueError("No valid configurations found!")
        
        # Convert to DataFrame for easier filtering (or use list comprehension)
        df = pd.DataFrame(results)
        
        #print(f"Total configurations: {len(df)}")
        #print(f"P-value range: {df['p_value'].min():.6f} to {df['p_value'].max():.6f}")
        
        # Filter by your criteria
        target_results = df[(df['p_value'] > 0.9) & (df['p_value'] < 1.0)]
        
        if len(target_results) > 0:
            # Select row with highest u value
            best_row = target_results.loc[target_results['u'].idxmax()]
            method = "highest_u_p_0.8_to_1"
        else:
            # Relax criteria
            relaxed_results = df[df['p_value'] > 0.7]
            if len(relaxed_results) > 0:
                best_row = relaxed_results.loc[relaxed_results['u'].idxmax()]
                method = "highest_u_p_0.5_to_1_relaxed"
            else:
                # Fallback
                best_row = df.loc[df['p_value'].idxmax()]
                method = "fallback_highest_p"
        
        definitive_params = {
            'u': best_row['u'], 'k': best_row['k'], 's': best_row['s'],
            'A2': best_row['A2'], 'p': best_row['p_value'],
            'start_index': best_row['start_index'], 'method': method
        }
        
        k_0 = definitive_params['k']
        s_0 = definitive_params['s']
        u_0 = definitive_params['u']
        A_0 = definitive_params['A2']
        p_0 = definitive_params['p']
        
        # Print final selection
        #print(f"\n FINAL SELECTION:")
        #print(f"   Start index: {selected_start_idx}")
        print(f"   u = {u_0:.6f}, A² = {A_0:.6f}, p-value = {p_0:.6f}")

        
        # Quality assessment
        if p_0 > 0.9:
            quality = " Excellent fit"
        elif p_0 > 0.5:
            quality = " Good fit"
        elif p_0 > 0.10:
            quality = " Acceptable  fit"
        else:
            quality = " Poor fit"
        print(f'Quality of fitness: {quality} \n')

        x_fit = np.linspace(min(sorted_picks_norp), max(sorted_picks_norp), 1000)


        
    # print(f'final Threshold: {u_0}, p-val: {p_0}')
        
        
        # Calculate GPD PDF and CDF using final parameters
        #pdf_pareto = gpd_pdf(x_fit, k_0, s_0, u_0)
        #cdf_pareto = gpd_cdf(x_fit, k_0, s_0, u_0)

        # Create figure with three subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # --- Plot 1: PDF Comparison (Histogram + Fitted GPD) ---
        ax1.hist(sorted_picks, bins=nbins*2, color='navy', 
                histtype='stepfilled', alpha=0.4, density=True, label='Data histogram')
        ax1.axvline(x=u_0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Threshold = {u_0:.2f} nT')

        ax1.set_title(f"{st} - GPD Fit (PDF)")
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)

        # Add goodness-of-fit info to CDF plot
        fit_text = f'Goodness-of-fit: A²={A_0:.3f}, p={p_0:.6f}'
        if p_0 > 0.9:
            fit_quality = 'Excellent fit'
        elif p_0 > 0.5:
            fit_quality = 'good fit' 
        elif p_0 > 0.1:
            fit_quality = 'acceptable fit'
        else:
            fit_quality = 'Poor fit'

        plt.text(0.1, 0.8, f'Fitness Quality: {quality}', horizontalalignment='center',
        verticalalignment='center', transform=ax2.transAxes)

        # --- Plot 2: Empirical CDF ---
        ax2.plot(np.sort(sorted_picks_norp), cdf, 'b-', linewidth=2, label='Empirical CDF')
        ax2.axvline(x=u_0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Threshold = {u_0:.2f} nT')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('CDF')
        ax2.set_title('Empirical Cumulative Distribution')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.3)




        plt.tight_layout()
        plt.savefig(f'/home/isaac/rutpy/gicsOutput/gic_dist/CDF_{st}_{i_date}_{f_date}.png', dpi=300)
        plt.close()
        threshold = u_0




    return threshold


def calculate_fwhm(x_data, y_data):
    """
    Calculates the FWHM of a single peak in sampled data.
    Assumes a single, well-defined peak.
    """
    half_max = np.max(y_data) / 2.0
    # Find the indices where the data crosses the half-maximum value
    signs = np.sign(np.add(y_data, -half_max))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]

    if zero_crossings_i.size < 2:
        return None # FWHM not well-defined for this data

    # Interpolate to find the exact x values at half maximum
    left_x = x_data[zero_crossings_i[0]] + (x_data[zero_crossings_i[0]+1] - x_data[zero_crossings_i[0]]) * ((half_max - y_data[zero_crossings_i[0]]) / (y_data[zero_crossings_i[0]+1] - y_data[zero_crossings_i[0]]))
    right_x = x_data[zero_crossings_i[1]] + (x_data[zero_crossings_i[1]+1] - x_data[zero_crossings_i[1]]) * ((half_max - y_data[zero_crossings_i[1]]) / (y_data[zero_crossings_i[1]+1] - y_data[zero_crossings_i[1]]))

    fwhm = right_x - left_x
    return fwhm

def calculate_cdf_fwhm_percentile(data):
    """
    Calcula FWHM para CDF usando percentiles
    """
    q1 = np.percentile(data, 25)   # Primer cuartil (25%)
    q3 = np.percentile(data, 75)   # Tercer cuartil (75%)
    fwhm_cdf = q3 - q1
    return fwhm_cdf, q1, q3

def gic_threshold(data, st, window_idx):
    
    data = np.array(data).flatten()
    ndata = len(data)

    data = data[~np.isnan(data)]
    
    
    nbins = nbin_compute(data, ndata)

    frequencies, bin_edges = np.histogram(data, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = np.arange(1, len(data)+1) / len(data)    
    fit_y, params = gaussian_fit(bin_centers, frequencies)
    H_fit, A_fit, x0_fit, sigma_fit = params

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Histograma con ajuste gaussiano
    ax1.bar(bin_centers, frequencies, width=bin_edges[1]-bin_edges[0], 
            alpha=0.6, color='navy')
    ax1.plot(bin_centers, fit_y, 'r-', linewidth=2, label='Gaussian fit')
    
    
    FWHM = calculate_fwhm(bin_centers, fit_y)
    #FWHM = (2*np.sqrt(2*np.log(2)))*sigma_fit
    
    
    ax1.axvline(x=x0_fit, color='darkorange', linestyle='-', linewidth=2, label=f'Media ($\mu$) = {x0_fit:.3f}')
    ax1.axvline(x=x0_fit - FWHM/2, color='darkorange', linestyle='--', linewidth=1.5, label=f'$FWHM = {FWHM:.3f}$')
    ax1.axvline(x=x0_fit + FWHM/2, color='darkorange', linestyle='--', linewidth=1.5)     
    
    ax1.set_xlabel('GIC [A]')
    ax1.set_ylabel('Probability Density')
    #ax1.set_title(f'GICs Distribution - {st}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: CDF (Función de Distribución Acumulada)
    
    data_sorted = np.sort(np.abs(data))
    ax2.plot(data_sorted, cdf, color='navy', linewidth=2)
  
    x_fit = data_sorted
    fit_y, sigma_fit, offset_fit = fit_half_normal_cdf(x_fit, cdf)    
    idx_95 = np.argmin(np.abs(fit_y - 0.95))
    valor_p = x_fit[idx_95]
    
    ax2.plot(x_fit, fit_y, 'r-', linewidth=2, label=f'Half-Normal fit')
    ax2.axvline(x=valor_p, color='green', linestyle='-', linewidth=2, 
            alpha=0.9, label=f'CDF 95% = {valor_p:.3f} A')

# Línea para μ + FWHM (2×FWHM desde la media)
    ax2.axvline(x=x0_fit + FWHM, color='green', linestyle=':', linewidth=2, 
            alpha=0.9, label=f'μ + FWHM = {x0_fit + FWHM:.3f} A')

# Línea para μ + FWHM/2
    ax2.axvline(x=x0_fit + FWHM/2, color='green', linestyle='--', linewidth=2, 
            alpha=0.9, label=f'μ + FWHM/2 = {x0_fit + FWHM/2:.3f} A')
    #ax2.hist(data, bins=nbins, density=True, cumulative=True, alpha=0.7, color='skyblue', label='CDF empírica')
    ax2.set_xlabel('GIC [A]')
    ax2.set_ylabel('Cumulative distribution')
    ax2.set_ylim(0,1.1)
   # ax2.set_title('Función de Distribución Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'GICs {st} Distribution during Window {window_idx}')
    plt.tight_layout()
    
    save_path = f'/home/isaac/gics_rv/fig/distributions/{st}/{st}_{window_idx}.minmax.png'
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(save_path):
        os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{save_path}', dpi=300)
    plt.close()
    
    return x0_fit, FWHM, valor_p