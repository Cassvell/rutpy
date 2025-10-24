import numpy as np
from lmoments3 import distr
import lmoments3 as lm
import kneed as kn
from scipy.stats import genpareto, kstest #anderson
import matplotlib.pyplot as plt
import pandas as pd

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


def get_threshold(picks, st, method):

    ndays = int(len(picks)/4)

    picks = np.array(picks)  

    picks = picks[~np.isnan(picks)]
    picks = np.unique(picks)
    
    hist, bins = np.histogram(picks, bins=ndays*2, density=True)    
    
    sorted_picks = np.sort(picks)

    
    nbins = int(len(sorted_picks) / 3)

    stddev_res = (np.array(sorted_picks).flatten())

    frequencies, bin_edges = np.histogram(stddev_res, bins=nbins*2, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cdf = np.arange(1, len(sorted_picks)+1) / len(sorted_picks) 
    
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
    
    elif method == 'GPD':
        definitive_params = final_params(sorted_picks, cdf,bound)

        p_0 = definitive_params['p']
        u_0 = definitive_params['u']
        A_0 = definitive_params['A2']
        
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
        x_fit = np.linspace(min(sorted_picks), max(sorted_picks), 1000)

        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # --- Plot 1: PDF Comparison (Histogram + Fitted GPD) ---
        #ax1.hist(sorted_picks, bins=nbins*2, color='navy', 
        #        histtype='stepfilled', alpha=0.4, density=True, label='Data histogram')
        #ax1.axvline(x=u_0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Threshold = {u_0:.2f} nT')

        #ax1.set_title(f"{st.upper()} - magnetic data distribution picks (PDF)")
        #ax1.set_ylabel('Density')
        #ax1.legend()
        #ax1.grid(True, which='both', alpha=0.3)

        # Add goodness-of-fit info to CDF plot
        fit_text = f'Goodness-of-fit: A²={A_0:.3f}, p={p_0:.6f}'
        if p_0 > 0.05:
            fit_quality = 'Good fit'
        elif p_0 > 0.01:
            fit_quality = 'Marginal fit'
        else:
            fit_quality = 'Poor fit'
        threshold = u_0
    #plt.text(0.1, 0.8, f'Fitness Quality: {quality}', horizontalalignment='center',
    # verticalalignment='center', transform=ax2.transAxes)

    # --- Plot 2: Empirical CDF ---
    #ax2.plot(np.sort(sorted_picks_norp), cdf, 'b-', linewidth=2, label='Empirical CDF')
    #ax2.axvline(x=u_0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=f'Threshold = {u_0:.2f} nT')
    #ax2.set_xlabel('Value')
    #ax2.set_ylabel('CDF')
    #ax2.set_title('Empirical Cumulative Distribution')
    #ax2.legend()
    #ax2.grid(True, which='both', alpha=0.3)




    #plt.tight_layout()
    #plt.savefig(f'/home/isaac/MEGA/posgrado/doctorado/semestre5/gpd_magdata/CDF_{st}_StPatrick_GPD.png', dpi=300)
    #plt.show()


    return threshold
###############################################################################
###############################################################################
#generates an array of variation picks    
import numpy as np
import sys

def med_IQR(data, tw, tw_pick, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()

    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / tw)
        hourly = []
        
        for i in range(hourly_sample):
            current_window = data[i * tw : (i + 1) * tw]
            
            if len(current_window) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
            
            if non_nan_ratio > 0.9:
                QR1_hr = np.nanquantile(current_window, 0.25)
                QR3_hr = np.nanquantile(current_window, 0.75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            
            hourly.append(iqr_hr)
        
        return hourly
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanstd(trihourly_window)  # Standard deviation of the window
        else:
            stdev = np.nan  # If the window is incomplete (less than 3 data points)
        trihourly_stdev.append(stdev)

    
    daily = []
    
    # For each day, we pick the maximum IQR or standard deviation based on tw_pick
    for i in range(int(24 / tw_pick) * ndays):        
        if method == 'iqr':
            iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.90:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.9:
                iqr_picks = np.nanmedian(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)

def max_IQR(data, tw, tw_pick, method='iqr'):
    ndata = len(data)
    ndays = int(ndata / 1440)

    if tw_pick == 0 or 24 % tw_pick != 0:
        print('Error: Please enter a time window in hours, divisor of 24 hours.')
        sys.exit()

    def hourly_IQR(data):
        ndata = len(data)
        hourly_sample = int(ndata / tw)
        hourly = []
        
        for i in range(hourly_sample):
            current_window = data[i * tw : (i + 1) * tw]
            
            if len(current_window) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(current_window)) / len(current_window)
            
            if non_nan_ratio > 0.9:
                QR1_hr = np.nanquantile(current_window, 0.25)
                QR3_hr = np.nanquantile(current_window, 0.75)
                iqr_hr = QR3_hr - QR1_hr
            else:
                iqr_hr = np.nan
            
            hourly.append(iqr_hr)
        
        return hourly
    
    # Compute the hourly IQR
    hourly = hourly_IQR(data)


        # Compute trihourly standard deviation from the hourly output
    trihourly_stdev = []
    
    for i in range(0, len(hourly), 3):  # Step by 3 to group into trihourly windows
        trihourly_window = hourly[i:i+3]  # A trihourly window
        if len(trihourly_window) == 3:
            stdev = np.nanstd(trihourly_window)  # Standard deviation of the window
        else:
            stdev = np.nan  # If the window is incomplete (less than 3 data points)
        trihourly_stdev.append(stdev)

    
    daily = []
    
    # For each day, we pick the maximum IQR or standard deviation based on tw_pick
    for i in range(int(24 / tw_pick) * ndays):        
        if method == 'iqr':
            iqr_mov = hourly[i * tw_pick : (i + 1) * tw_pick]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.90:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan
            
        elif method == 'stddev':
            iqr_mov = trihourly_stdev[i * int(tw_pick/3) : (i + 1) * int(tw_pick/3)]
            if len(iqr_mov) == 0:
                continue  # Skip empty windows
            
            non_nan_ratio = np.sum(~np.isnan(iqr_mov)) / len(iqr_mov)
            
            if non_nan_ratio > 0.9:
                iqr_picks = np.nanmax(iqr_mov)  # Pick the max value for IQR or stdev
            else:
                iqr_picks = np.nan                   
            
        daily.append(iqr_picks)
        
    return np.array(daily)