
import matplotlib.pyplot as plt
from gicdproc import  process_station_data
from timeit import default_timer as timer

import pandas as pd
import os.path

import numpy as np
from datetime import datetime, timedelta
from calc_daysdiff import calculate_days_difference
from ts_acc import mz_score
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.dates as mdates
#from corr_offset import corr_offset
import sys
import os
import lmoments3 as lm
from scipy.stats import genpareto
from scipy.stats import percentileofscore
import math


def gpd_pdf(x, k, sigma, mu):
    """
    Generalized Pareto Distribution PDF
    
    Parameters:
    x: array-like, input values
    k: shape parameter
    sigma: scale parameter (sigma > 0)
    mu: location parameter
    """
    x = np.array(x)
    z = (x - mu) / sigma
    
    if k == 0:
        # Exponential special case
        return (1/sigma) * np.exp(-z)
    else:
        # General case (k != 0)
        # Check support: 1 + k*z > 0
        mask = (1 + k*z) > 0
        result = np.zeros_like(x)
        result[mask] = (1/sigma) * (1 + k*z[mask])**(-1 - 1/k)
        return result

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

def check_data_type(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return 0
    elif isinstance(data, np.ndarray):
        return 1
    else:
        return "Neither Pandas nor NumPy"

def threshold(picks, i_date, f_date, st):
    sorted_picks = np.array(picks)
    sorted_picks = np.sort(sorted_picks)

    sorted_picks_norp = np.unique(sorted_picks)
    nbins = int(len(sorted_picks) / 3)

    stddev_res = (np.array(sorted_picks_norp).flatten())

    frequencies, bin_edges = np.histogram(stddev_res, bins=nbins*2, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = np.arange(1, len(sorted_picks_norp)+1) / len(sorted_picks_norp)

    #k, s, u = GPD_params(sorted_picks_norp)
    
    



   # optimal_mu, optimal_p, result = find_optimal_mu_constrained(sorted_picks_norp, k, s, u, n_simulations=500)
    bound =   0.95
    idx = np.searchsorted(cdf, bound)
    
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
    target_results = df[(df['p_value'] > 0.8) & (df['p_value'] < 1.0)]
    
    if len(target_results) > 0:
        # Select row with highest u value
        best_row = target_results.loc[target_results['u'].idxmax()]
        method = "highest_u_p_0.8_to_1"
    else:
        # Relax criteria
        relaxed_results = df[df['p_value'] > 0.6]
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
    print(f"\n FINAL SELECTION:")
    #print(f"   Start index: {selected_start_idx}")
    print(f"   u = {u_0:.6f}, A² = {A_0:.6f}, p-value = {p_0:.6f}")

    
    # Quality assessment
    if p_0 > 0.8:
        quality = " Excellent fit"
    elif p_0 > 0.6:
        quality = " Good fit"
    elif p_0 > 0.05:
        quality = " Marginal fit"
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
    if p_0 > 0.05:
        fit_quality = 'Good fit'
    elif p_0 > 0.01:
        fit_quality = 'Marginal fit'
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
    plt.show()


    return u_0



