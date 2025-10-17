import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
path = f'/home/isaac/MEGA/posgrado/doctorado/semestre5/'

def linear_func(x, a, b):
    """Linear function: a*x + b"""
    return a * x + b

def quadratic_func(x, a, b, c):
    """Quadratic function: a*x² + b*x + c"""
    return a * x**2 + b * x + c

df = pd.read_csv(f'{path}gics_tabla.csv', sep=',', header = 1)
df = df.replace('n', np.nan)  
df = df.replace('n/a', np.nan)    

df['LAVfit'] = df.iloc[:,8]
df['QROfit'] = df.iloc[:,11]
df['RMYfit'] = df.iloc[:,14]
df['MZTfit'] = df.iloc[:,17]

gpd_lav = df.iloc[:,7]
gpd_qro = df.iloc[:,10]
gpd_rmy = df.iloc[:,13]
gpd_mzt = df.iloc[:,16]

s2_lav = df.iloc[:,9]
s2_qro = df.iloc[:,12]
s2_rmy = df.iloc[:,15]
s2_mzt = df.iloc[:,18]

s3_lav = df.iloc[:,9]*(3/2)
s3_qro = df.iloc[:,12]*(3/2)
s3_rmy = df.iloc[:,15]*(3/2)
s3_mzt = df.iloc[:,18]*(3/2)

peak_lav = df.iloc[:,20]
peak_qro = df.iloc[:,21]
peak_rmy = df.iloc[:,22]
peak_mzt = df.iloc[:,23]

dH_min = df.iloc[:,2]
K_max = df.iloc[:,5]



def plot_threshold_dispersion(axes, threshold_data, k_data, color, station_name, threshold_type, station_idx):
    """
    Create dispersion plots for threshold values vs K_max
    """
    # Convert to numpy and remove NaNs
    threshold_np = np.array(threshold_data)
    k_np = np.array(k_data)
    
    mask = ~np.isnan(threshold_np) & ~np.isnan(k_np)
    clean_threshold = threshold_np[mask]
    clean_k = k_np[mask]
    
    if len(clean_threshold) < 5:
        print(f"Warning: Not enough data for {station_name} {threshold_type} (n={len(clean_threshold)})")
        return None, None
    
    # Get axis for this station and threshold type
    ax = axes[station_idx]
    
    # Create scatter plot
    ax.scatter(clean_threshold, clean_k, color=color, alpha=0.7, s=50, 
              label=f'{threshold_type} (n={len(clean_threshold)})')
    
    # Fit both linear and quadratic functions
    try:
        # Linear fit
        popt_lin, _ = curve_fit(linear_func, clean_threshold, clean_k)
        
        # Quadratic fit
        popt_quad, _ = curve_fit(quadratic_func, clean_threshold, clean_k, 
                                p0=[0.001, popt_lin[0], popt_lin[1]], maxfev=5000)
        
        # Generate fit curves
        x_fit = np.linspace(clean_threshold.min(), clean_threshold.max(), 200)
        
        # Calculate curves
        y_lin = linear_func(x_fit, *popt_lin)
        y_quad = quadratic_func(x_fit, *popt_quad)
        
        # Calculate R² values
        def calculate_r2(x, y, popt, func):
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        r2_lin = calculate_r2(clean_threshold, clean_k, popt_lin, linear_func)
        r2_quad = calculate_r2(clean_threshold, clean_k, popt_quad, quadratic_func)
        
        # Plot fits
        ax.plot(x_fit, y_lin, '--', color='black', linewidth=2, alpha=0.8,
               label=f'Linear (R²={r2_lin:.3f})')
        ax.plot(x_fit, y_quad, '-', color=color, linewidth=2,
               label=f'Quadratic (R²={r2_quad:.3f})')
        
        # Add statistics to plot
        stats_text = f'Linear R² = {r2_lin:.3f}\nQuadratic R² = {r2_quad:.3f}\nImprovement = {r2_quad - r2_lin:+.3f}'
        
        ax.text(0.05, 0.15, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        return (popt_lin, r2_lin), (popt_quad, r2_quad)
        
    except Exception as e:
        print(f"Error fitting for {station_name} {threshold_type}: {e}")
        return None, None

# Create figures for each threshold type
fig_gpd, axes_gpd = plt.subplots(2, 2, figsize=(15, 12))
fig_s2, axes_s2 = plt.subplots(2, 2, figsize=(15, 12))
fig_s3, axes_s3 = plt.subplots(2, 2, figsize=(15, 12))

fig_gpd.suptitle(r'$K_{max}$ vs GPD Threshold', fontsize=16)
fig_s2.suptitle(r'$K_{max}$ vs S2 Threshold', fontsize=16)
fig_s3.suptitle(r'$K_{max}$ vs S3 Threshold', fontsize=16)

axes_gpd = axes_gpd.flatten()
axes_s2 = axes_s2.flatten()
axes_s3 = axes_s3.flatten()

# Stations data with all threshold types
stations_data = [
    {'name': 'LAV', 'color': 'blue', 'idx': 0, 
     'gpd': gpd_lav, 's2': s2_lav, 's3': s3_lav},
    {'name': 'QRO', 'color': 'orange', 'idx': 1,
     'gpd': gpd_qro, 's2': s2_qro, 's3': s3_qro},
    {'name': 'RMY', 'color': 'green', 'idx': 2,
     'gpd': gpd_rmy, 's2': s2_rmy, 's3': s3_rmy},
    {'name': 'MZT', 'color': 'purple', 'idx': 3,
     'gpd': gpd_mzt, 's2': s2_mzt, 's3': s3_mzt}
]

# Store fitted parameters
fitted_params = {}

# Create dispersion plots for each station and threshold type
for station in stations_data:
    station_name = station['name']
    color = station['color']
    idx = station['idx']
    
    fitted_params[station_name] = {}
    
    # GPD vs K_max
    lin_gpd, quad_gpd = plot_threshold_dispersion(axes_gpd, station['gpd'], K_max, color, 
                                                station_name, 'GPD', idx)
    fitted_params[station_name]['GPD'] = {'linear': lin_gpd, 'quadratic': quad_gpd}
    
    # S2 vs K_max
    lin_s2, quad_s2 = plot_threshold_dispersion(axes_s2, station['s2'], K_max, color,
                                              station_name, 'S2', idx)
    fitted_params[station_name]['S2'] = {'linear': lin_s2, 'quadratic': quad_s2}
    
    # S3 vs K_max
    lin_s3, quad_s3 = plot_threshold_dispersion(axes_s3, station['s3'], K_max, color,
                                              station_name, 'S3', idx)
    fitted_params[station_name]['S3'] = {'linear': lin_s3, 'quadratic': quad_s3}
    
    # Format GPD subplot
    axes_gpd[idx].set_xlabel('GPD Threshold [nT]')
    axes_gpd[idx].set_ylabel(r'$K_{max}$')
    axes_gpd[idx].grid(True, alpha=0.3)
    axes_gpd[idx].legend()
    axes_gpd[idx].set_ylim(0, 9)
    axes_gpd[idx].set_title(f'{station_name}')
    
    # Format S2 subplot
    axes_s2[idx].set_xlabel('S2 Threshold [nT]')
    axes_s2[idx].set_ylabel(r'$K_{max}$')
    axes_s2[idx].grid(True, alpha=0.3)
    axes_s2[idx].legend()
    axes_s2[idx].set_ylim(0, 9)
    axes_s2[idx].set_title(f'{station_name}')
    
    # Format S3 subplot
    axes_s3[idx].set_xlabel('S3 Threshold [nT]')
    axes_s3[idx].set_ylabel(r'$K_{max}$')
    axes_s3[idx].grid(True, alpha=0.3)
    axes_s3[idx].legend()
    axes_s3[idx].set_ylim(0, 9)
    axes_s3[idx].set_title(f'{station_name}')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Print fitted parameters
print("Linear and Quadratic Fit Parameters for K_max vs Threshold Values:")
print("="*90)

for station_name in ['LAV', 'QRO', 'RMY', 'MZT']:
    station_fits = fitted_params.get(station_name, {})
    
    print(f"\n{station_name}:")
    for threshold_type in ['GPD', 'S2', 'S3']:
        fits = station_fits.get(threshold_type, {})
        lin_fit = fits.get('linear')
        quad_fit = fits.get('quadratic')
        
        if lin_fit is not None and quad_fit is not None:
            popt_lin, r2_lin = lin_fit
            popt_quad, r2_quad = quad_fit
            
            print(f"  {threshold_type}:")
            print(f"    Linear:    K_max = {popt_lin[0]:.4f}·Threshold + {popt_lin[1]:.3f}  (R² = {r2_lin:.3f})")
            print(f"    Quadratic: K_max = {popt_quad[0]:.6f}·Threshold² + {popt_quad[1]:.4f}·Threshold + {popt_quad[2]:.3f}  (R² = {r2_quad:.3f})")
            print(f"    Improvement: {r2_quad - r2_lin:+.3f}")



fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Threshold vs Peak', fontsize=16)

def plot_peak_comparisons_with_regression(ax, gpd, s3, peak, station_name):
    # Remove NaN values
    mask_gpd = ~(np.isnan(gpd) | np.isnan(peak))
    mask_s3 = ~(np.isnan(s3) | np.isnan(peak))
    
    gpd_clean = gpd[mask_gpd]
    peak_gpd_clean = peak[mask_gpd]
    s3_clean = s3[mask_s3] 
    peak_s3_clean = peak[mask_s3]
    
    # Plot scatter points
    ax.scatter(gpd_clean, peak_gpd_clean, color='blue', alpha=0.7, 
               label='GPD vs Peak', s=50)
    ax.scatter(s3_clean, peak_s3_clean, color='red', alpha=0.5, 
               label='S3 vs Peak', s=50)
    
    # Fit and plot regression lines
    if len(gpd_clean) > 1:
        # GPD regression
        gpd_model = LinearRegression()
        gpd_model.fit(gpd_clean.values.reshape(-1, 1), peak_gpd_clean)
        gpd_line_x = np.linspace(gpd_clean.min(), gpd_clean.max(), 100)
        gpd_line_y = gpd_model.predict(gpd_line_x.reshape(-1, 1))
        ax.plot(gpd_line_x, gpd_line_y, 'blue', linestyle='-', alpha=0.8, linewidth=2)
    
    if len(s3_clean) > 1:
        # S3 regression
        s3_model = LinearRegression()
        s3_model.fit(s3_clean.values.reshape(-1, 1), peak_s3_clean)
        s3_line_x = np.linspace(s3_clean.min(), s3_clean.max(), 100)
        s3_line_y = s3_model.predict(s3_line_x.reshape(-1, 1))
        ax.plot(s3_line_x, s3_line_y, 'red', linestyle='-', alpha=0.8, linewidth=2)
    
    # Identity line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Absolute Peak')
    ax.set_title(f'{station_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Calculate and display statistics
    corr_gpd = np.corrcoef(gpd_clean, peak_gpd_clean)[0,1] if len(gpd_clean) > 1 else np.nan
    corr_s3 = np.corrcoef(s3_clean, peak_s3_clean)[0,1] if len(s3_clean) > 1 else np.nan
    
    textstr = f'n(GPD): {len(gpd_clean)}\nn(S3): {len(s3_clean)}\nCorr(GPD): {corr_gpd:.3f}\nCorr(S3): {corr_s3:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

# LAV plot
plot_peak_comparisons_with_regression(axes[0,0], gpd_lav, s3_lav, peak_lav, 'LAV')

# QRO plot
plot_peak_comparisons_with_regression(axes[0,1], gpd_qro, s3_qro, peak_qro, 'QRO')

# RMY plot
plot_peak_comparisons_with_regression(axes[1,0], gpd_rmy, s3_rmy, peak_rmy, 'RMY')

# MZT plot
plot_peak_comparisons_with_regression(axes[1,1], gpd_mzt, s3_mzt, peak_mzt, 'MZT')

plt.tight_layout()
plt.savefig('/home/isaac/MEGA/posgrado/doctorado/semestre5/peakvsthreshold.png', dpi=300)
plt.close()


# Create subplots - 2x2 grid for dH_min < -75
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle('Threshold vs Peak (dH_min < -75 nT)', fontsize=16)

# Create subplots - 2x2 grid for dH_min < -100
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Threshold vs Peak (dH_min < -100 nT)', fontsize=16)

def plot_filtered_by_dH(ax, gpd, s3, peak, dH_min_vals, station_name, dH_threshold):
    # Create mask for dH_min condition
    dH_mask = dH_min_vals < dH_threshold
    
    # Remove NaN values AND apply dH filter
    mask_gpd = ~(np.isnan(gpd) | np.isnan(peak)) & dH_mask
    mask_s3 = ~(np.isnan(s3) | np.isnan(peak)) & dH_mask
    
    gpd_clean = gpd[mask_gpd]
    peak_gpd_clean = peak[mask_gpd]
    s3_clean = s3[mask_s3] 
    peak_s3_clean = peak[mask_s3]
    
    # Plot scatter points
    ax.scatter(gpd_clean, peak_gpd_clean, color='blue', alpha=0.7, 
               label='GPD vs Peak', s=50)
    ax.scatter(s3_clean, peak_s3_clean, color='red', alpha=0.5, 
               label='S3 vs Peak', s=50)
    
    # Fit and plot regression lines
    if len(gpd_clean) > 1:
        # GPD regression
        gpd_model = LinearRegression()
        gpd_model.fit(gpd_clean.values.reshape(-1, 1), peak_gpd_clean)
        gpd_line_x = np.linspace(gpd_clean.min(), gpd_clean.max(), 100)
        gpd_line_y = gpd_model.predict(gpd_line_x.reshape(-1, 1))
        ax.plot(gpd_line_x, gpd_line_y, 'blue', linestyle='-', alpha=0.8, linewidth=2)
    
    if len(s3_clean) > 1:
        # S3 regression
        s3_model = LinearRegression()
        s3_model.fit(s3_clean.values.reshape(-1, 1), peak_s3_clean)
        s3_line_x = np.linspace(s3_clean.min(), s3_clean.max(), 100)
        s3_line_y = s3_model.predict(s3_line_x.reshape(-1, 1))
        ax.plot(s3_line_x, s3_line_y, 'red', linestyle='-', alpha=0.8, linewidth=2)
    
    # Identity line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Absolute Peak')
    ax.set_title(f'{station_name} (dH < {dH_threshold} nT)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Calculate and display statistics
    corr_gpd = np.corrcoef(gpd_clean, peak_gpd_clean)[0,1] if len(gpd_clean) > 1 else np.nan
    corr_s3 = np.corrcoef(s3_clean, peak_s3_clean)[0,1] if len(s3_clean) > 1 else np.nan
    
    textstr = f'n(GPD): {len(gpd_clean)}\nn(S3): {len(s3_clean)}\nCorr(GPD): {corr_gpd:.3f}\nCorr(S3): {corr_s3:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

# Plot for dH_min < -75
plot_filtered_by_dH(axes1[0,0], gpd_lav, s3_lav, peak_lav, dH_min, 'LAV', -75)
plot_filtered_by_dH(axes1[0,1], gpd_qro, s3_qro, peak_qro, dH_min, 'QRO', -75)
plot_filtered_by_dH(axes1[1,0], gpd_rmy, s3_rmy, peak_rmy, dH_min, 'RMY', -75)
plot_filtered_by_dH(axes1[1,1], gpd_mzt, s3_mzt, peak_mzt, dH_min, 'MZT', -75)

# Plot for dH_min < -100
plot_filtered_by_dH(axes2[0,0], gpd_lav, s3_lav, peak_lav, dH_min, 'LAV', -100)
plot_filtered_by_dH(axes2[0,1], gpd_qro, s3_qro, peak_qro, dH_min, 'QRO', -100)
plot_filtered_by_dH(axes2[1,0], gpd_rmy, s3_rmy, peak_rmy, dH_min, 'RMY', -100)
plot_filtered_by_dH(axes2[1,1], gpd_mzt, s3_mzt, peak_mzt, dH_min, 'MZT', -100)

plt.tight_layout()
plt.close()


# Create subplots - 2x2 grid for K_max <= 5
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
fig3.suptitle('Threshold vs Peak (K_max ≤ 5)', fontsize=16)

# Create subplots - 2x2 grid for K_max between 6-9
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))
fig4.suptitle('Threshold vs Peak (6 ≤ K_max ≤ 9)', fontsize=16)

def plot_filtered_by_Kmax(ax, gpd, s3, peak, K_max_vals, station_name, K_condition):
    # Create mask for K_max condition
    if K_condition == "leq5":
        K_mask = K_max_vals <= 5
        title_suffix = "K_max ≤ 5"
    elif K_condition == "between6_9":
        K_mask = (K_max_vals >= 6) & (K_max_vals <= 9)
        title_suffix = "6 ≤ K_max ≤ 9"
    
    # Remove NaN values AND apply K_max filter
    mask_gpd = ~(np.isnan(gpd) | np.isnan(peak)) & K_mask
    mask_s3 = ~(np.isnan(s3) | np.isnan(peak)) & K_mask
    
    gpd_clean = gpd[mask_gpd]
    peak_gpd_clean = peak[mask_gpd]
    s3_clean = s3[mask_s3] 
    peak_s3_clean = peak[mask_s3]
    
    # Plot scatter points
    ax.scatter(gpd_clean, peak_gpd_clean, color='blue', alpha=0.7, 
               label='GPD vs Peak', s=50)
    ax.scatter(s3_clean, peak_s3_clean, color='red', alpha=0.5, 
               label='S3 vs Peak', s=50)
    
    # Fit and plot regression lines
    if len(gpd_clean) > 1:
        # GPD regression
        gpd_model = LinearRegression()
        gpd_model.fit(gpd_clean.values.reshape(-1, 1), peak_gpd_clean)
        gpd_line_x = np.linspace(gpd_clean.min(), gpd_clean.max(), 100)
        gpd_line_y = gpd_model.predict(gpd_line_x.reshape(-1, 1))
        ax.plot(gpd_line_x, gpd_line_y, 'blue', linestyle='-', alpha=0.8, linewidth=2)
    
    if len(s3_clean) > 1:
        # S3 regression
        s3_model = LinearRegression()
        s3_model.fit(s3_clean.values.reshape(-1, 1), peak_s3_clean)
        s3_line_x = np.linspace(s3_clean.min(), s3_clean.max(), 100)
        s3_line_y = s3_model.predict(s3_line_x.reshape(-1, 1))
        ax.plot(s3_line_x, s3_line_y, 'red', linestyle='-', alpha=0.8, linewidth=2)
    
    # Identity line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Absolute Peak')
    ax.set_title(f'{station_name} ({title_suffix})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Calculate and display statistics
    corr_gpd = np.corrcoef(gpd_clean, peak_gpd_clean)[0,1] if len(gpd_clean) > 1 else np.nan
    corr_s3 = np.corrcoef(s3_clean, peak_s3_clean)[0,1] if len(s3_clean) > 1 else np.nan
    
    textstr = f'n(GPD): {len(gpd_clean)}\nn(S3): {len(s3_clean)}\nCorr(GPD): {corr_gpd:.3f}\nCorr(S3): {corr_s3:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

# Plot for K_max <= 5
plot_filtered_by_Kmax(axes3[0,0], gpd_lav, s3_lav, peak_lav, K_max, 'LAV', "leq5")
plot_filtered_by_Kmax(axes3[0,1], gpd_qro, s3_qro, peak_qro, K_max, 'QRO', "leq5")
plot_filtered_by_Kmax(axes3[1,0], gpd_rmy, s3_rmy, peak_rmy, K_max, 'RMY', "leq5")
plot_filtered_by_Kmax(axes3[1,1], gpd_mzt, s3_mzt, peak_mzt, K_max, 'MZT', "leq5")

# Plot for K_max between 6-9
plot_filtered_by_Kmax(axes4[0,0], gpd_lav, s3_lav, peak_lav, K_max, 'LAV', "between6_9")
plot_filtered_by_Kmax(axes4[0,1], gpd_qro, s3_qro, peak_qro, K_max, 'QRO', "between6_9")
plot_filtered_by_Kmax(axes4[1,0], gpd_rmy, s3_rmy, peak_rmy, K_max, 'RMY', "between6_9")
plot_filtered_by_Kmax(axes4[1,1], gpd_mzt, s3_mzt, peak_mzt, K_max, 'MZT', "between6_9")

plt.tight_layout()
plt.show()


mask_lav = (df['LAVfit'] != 'p') & (df['LAVfit'].notna())
mask_qro = (df['QROfit'] != 'p') & (df['QROfit'].notna())
mask_rmy = (df['RMYfit'] != 'p') & (df['RMYfit'].notna())
mask_mzt = (df['MZTfit'] != 'p') & (df['MZTfit'].notna())

# Apply masks to filter data
gpd_lav_filtered = gpd_lav[mask_lav]
s2_lav_filtered = s2_lav[mask_lav]
s3_lav_filtered = s3_lav[mask_lav]

gpd_qro_filtered = gpd_qro[mask_qro]
s2_qro_filtered = s2_qro[mask_qro]
s3_qro_filtered = s3_qro[mask_qro]

gpd_rmy_filtered = gpd_rmy[mask_rmy]
s2_rmy_filtered = s2_rmy[mask_rmy]
s3_rmy_filtered = s3_rmy[mask_rmy]

gpd_mzt_filtered = gpd_mzt[mask_mzt]
s2_mzt_filtered = s2_mzt[mask_mzt]
s3_mzt_filtered = s3_mzt[mask_mzt]

ppc_lav_s2 = (gpd_lav-s2_lav)/s2_lav
ppc_qro_s2 = (gpd_qro-s2_qro)/s2_qro
ppc_rmy_s2 = (gpd_rmy-s2_rmy)/s2_rmy
ppc_mzt_s2 = (gpd_mzt-s2_mzt)/s2_mzt


ppc_lav_s3 = (gpd_lav-s3_lav)/s3_lav
ppc_qro_s3 = (gpd_qro-s3_qro)/s3_qro
ppc_rmy_s3 = (gpd_rmy-s3_rmy)/s3_rmy
ppc_mzt_s3 = (gpd_mzt-s3_mzt)/s3_mzt

ppc_lav_peak = (peak_lav-s2_lav)/s2_lav
ppc_qro_peak = (peak_qro-s2_qro)/s2_qro
ppc_rmy_peak = (peak_rmy-s2_rmy)/s2_rmy
ppc_mzt_peak= (peak_mzt-s2_mzt)/s2_mzt


ppc_lav_peak3 = (peak_lav-s3_lav)/s3_lav
ppc_qro_peak3 = (peak_qro-s3_qro)/s3_qro
ppc_rmy_peak3 = (peak_rmy-s3_rmy)/s3_rmy
ppc_mzt_peak3 = (peak_mzt-s3_mzt)/s3_mzt


def cdf_plot(ax, data_arrays, labels, colors, title):
    """
    Plot CDF for multiple data arrays on the same axis
    """
    for data, label, color in zip(data_arrays, labels, colors):
        # Remove NaN values and sort
        clean_data = data[~np.isnan(data)] * 100
        if len(clean_data) > 0:
            sorted_data = np.sort(clean_data)
            # Calculate CDF values
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            # Plot CDF
            ax.plot(sorted_data, cdf, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Increment [%]')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CDF for ppc differential Values', fontsize=16)

# Panel 1: GPD vs S2 PPC
cdf_plot(axes[0,0], 
         [ppc_lav_s2, ppc_qro_s2, ppc_rmy_s2, ppc_mzt_s2],
         ['LAV GPD-S2', 'QRO GPD-S2', 'RMY GPD-S2', 'MZT GPD-S2'],
         ['blue', 'orange', 'green', 'purple'],
         'GPD vs S2 PPC difference Distributions')

# Panel 2: GPD vs S3 PPC
cdf_plot(axes[0,1], 
         [ppc_lav_s3, ppc_qro_s3, ppc_rmy_s3, ppc_mzt_s3],
         ['LAV GPD-S3', 'QRO GPD-S3', 'RMY GPD-S3', 'MZT GPD-S3'],
         ['blue', 'orange', 'green', 'purple'],
         'GPD vs S3 PPC difference Distributions')

# Panel 3: Peak vs S2 PPC
cdf_plot(axes[1,0], 
         [ppc_lav_peak, ppc_qro_peak, ppc_rmy_peak, ppc_mzt_peak],
         ['LAV Peak-S2', 'QRO Peak-S2', 'RMY Peak-S2', 'MZT Peak-S2'],
         ['blue', 'orange', 'green', 'purple'],
         'Peak vs S2 PPC difference Distributions')

# Panel 4: Peak vs S3 PPC
cdf_plot(axes[1,1], 
         [ppc_lav_peak3, ppc_qro_peak3, ppc_rmy_peak3, ppc_mzt_peak3],
         ['LAV Peak-S3', 'QRO Peak-S3', 'RMY Peak-S3', 'MZT Peak-S3'],
         ['blue', 'orange', 'green', 'purple'],
         'Peak vs S3 PPC difference Distributions')

plt.tight_layout()
plt.savefig('/home/isaac/MEGA/posgrado/doctorado/semestre5/cdf_difference.png', dpi=300)
plt.close()


def fit_normal_cdf(x, mu, sigma):
    """Normal CDF function for fitting"""
    return stats.norm.cdf(x, loc=mu, scale=sigma)

def fit_cdf_with_confidence(ax, data, color, label, station_name):
    """
    Fit CDF to data and plot with confidence intervals
    """
    # Convert to numpy and remove NaNs
    data_np = np.array(data)
    clean_data = data_np[~np.isnan(data_np)] * 100  # Convert to percentage
    
    if len(clean_data) < 10:
        print(f"Warning: Not enough data for {station_name} (n={len(clean_data)})")
        return None, None
    
    # Sort data for CDF
    sorted_data = np.sort(clean_data)
    cdf_empirical = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Fit normal distribution to the data
    try:
        # Initial guess for parameters
        mu_guess = np.median(clean_data)
        sigma_guess = np.std(clean_data)
        
        # Fit normal CDF
        popt, pcov = curve_fit(fit_normal_cdf, sorted_data, cdf_empirical, 
                              p0=[mu_guess, sigma_guess], maxfev=5000)
        mu_fit, sigma_fit = popt
        
        # Generate fitted CDF
        x_fit = np.linspace(sorted_data.min(), sorted_data.max(), 200)
        y_fit = fit_normal_cdf(x_fit, mu_fit, sigma_fit)
        
        # Calculate confidence intervals using bootstrap
        n_bootstrap = 1000
        bootstrap_params = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
            bootstrap_sorted = np.sort(bootstrap_sample)
            bootstrap_cdf = np.arange(1, len(bootstrap_sorted) + 1) / len(bootstrap_sorted)
            
            try:
                bootstrap_popt, _ = curve_fit(fit_normal_cdf, bootstrap_sorted, bootstrap_cdf, 
                                            p0=[mu_fit, sigma_fit], maxfev=1000)
                bootstrap_params.append(bootstrap_popt)
            except:
                continue
        
        bootstrap_params = np.array(bootstrap_params)
        
        if len(bootstrap_params) > 0:
            # Calculate confidence intervals
            y_bootstrap = []
            for params in bootstrap_params:
                y_bootstrap.append(fit_normal_cdf(x_fit, params[0], params[1]))
            
            y_bootstrap = np.array(y_bootstrap)
            lower_ci = np.percentile(y_bootstrap, 2.5, axis=0)
            upper_ci = np.percentile(y_bootstrap, 97.5, axis=0)
            
            # Plot confidence interval
            ax.fill_between(x_fit, lower_ci, upper_ci, color=color, alpha=0.3, 
                           label=f'(95% CI)')
        
        # Plot empirical CDF
        ax.plot(sorted_data, cdf_empirical, 'o', color=color, markersize=3, alpha=0.7, 
               label=f' (Empirical)')
        
        # Plot fitted CDF
        ax.plot(x_fit, y_fit, '-', color=color, linewidth=2, 
               label=f'(Fitted: μ={mu_fit:.1f}%, σ={sigma_fit:.1f}%)')
        
        # Add statistics
        ax.text(0.05, 0.95, f'n = {len(clean_data)}\nμ = {mu_fit:.1f}%\nσ = {sigma_fit:.1f}%', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return mu_fit, sigma_fit
        
    except Exception as e:
        print(f"Error fitting CDF for {station_name}: {e}")
        # Plot just empirical CDF if fitting fails
        ax.plot(sorted_data, cdf_empirical, 'o-', color=color, markersize=3, linewidth=2,
               label=f'{label} (Empirical, n={len(clean_data)})')
        return None, None

# Create figure with 4 panels
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CDFs for GPD vs S2 PPC Difference', fontsize=16)

# Flatten axes for easy iteration
axes = axes.flatten()

# Stations data
stations_data = [
    (ppc_lav_s2, 'LAV', 'blue', axes[0]),
    (ppc_qro_s2, 'QRO', 'orange', axes[1]),
    (ppc_rmy_s2, 'RMY', 'green', axes[2]),
    (ppc_mzt_s2, 'MZT', 'purple', axes[3])
]

# Store fitted parameters
fitted_params = {}

# Fit and plot CDF for each station
for data, station_name, color, ax in stations_data:
    mu, sigma = fit_cdf_with_confidence(ax, data, color, station_name, station_name)
    fitted_params[station_name] = (mu, sigma)
    
    ax.set_xlabel('PPC Difference [%]')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{station_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-100, 200)  # Consistent x-axis limits

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()

# Print fitted parameters
print("Fitted Normal Distribution Parameters:")
print("="*50)
for station, (mu, sigma) in fitted_params.items():
    if mu is not None and sigma is not None:
        print(f"{station}: μ = {mu:.2f}%, σ = {sigma:.2f}%")


def linear_func(x, a, b):
    """Linear function: a*x + b"""
    return a * x + b

def quadratic_func(x, a, b, c):
    """Quadratic function: a*x² + b*x + c"""
    return a * x**2 + b * x + c

def plot_station_dispersion(axes_k, axes_dh, ppc_data, geo_data_k, geo_data_dh, color, station_name, station_idx):
    """
    Create dispersion plots for a single station with linear and quadratic fits
    """
    # Convert to numpy and remove NaNs
    ppc_np = np.array(ppc_data)
    k_np = np.array(geo_data_k)
    dh_np = np.array(geo_data_dh)
    
    mask = ~np.isnan(ppc_np)
    clean_ppc = ppc_np[mask] * 100  # Convert to percentage
    clean_k = k_np[mask]
    clean_dh = dh_np[mask]
    
    if len(clean_ppc) < 5:
        print(f"Warning: Not enough data for {station_name} (n={len(clean_ppc)})")
        return None, None, None, None
    
    # Get axes for this station
    ax_k = axes_k[station_idx]
    ax_dh = axes_dh[station_idx]
    
    # Create scatter plots
    ax_k.scatter(clean_ppc, clean_k, color=color, alpha=0.7, s=50, 
                label=f'{station_name} (n={len(clean_ppc)})')
    ax_dh.scatter(clean_ppc, clean_dh, color=color, alpha=0.7, s=50, 
                 label=f'{station_name} (n={len(clean_ppc)})')
    
    # Fit both linear and quadratic functions
    try:
        # K_max fits
        popt_lin_k, _ = curve_fit(linear_func, clean_ppc, clean_k)
        popt_quad_k, _ = curve_fit(quadratic_func, clean_ppc, clean_k, 
                                  p0=[0.001, popt_lin_k[0], popt_lin_k[1]], maxfev=5000)
        
        # dH_min fits
        popt_lin_dh, _ = curve_fit(linear_func, clean_ppc, clean_dh)
        popt_quad_dh, _ = curve_fit(quadratic_func, clean_ppc, clean_dh, 
                                   p0=[0.001, popt_lin_dh[0], popt_lin_dh[1]], maxfev=5000)
        
        # Generate fit curves
        x_fit = np.linspace(clean_ppc.min(), clean_ppc.max(), 200)
        
        # K_max curves
        y_lin_k = linear_func(x_fit, *popt_lin_k)
        y_quad_k = quadratic_func(x_fit, *popt_quad_k)
        
        # dH_min curves
        y_lin_dh = linear_func(x_fit, *popt_lin_dh)
        y_quad_dh = quadratic_func(x_fit, *popt_quad_dh)
        
        # Calculate R² values
        def calculate_r2(x, y, popt, func):
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        r2_lin_k = calculate_r2(clean_ppc, clean_k, popt_lin_k, linear_func)
        r2_quad_k = calculate_r2(clean_ppc, clean_k, popt_quad_k, quadratic_func)
        r2_lin_dh = calculate_r2(clean_ppc, clean_dh, popt_lin_dh, linear_func)
        r2_quad_dh = calculate_r2(clean_ppc, clean_dh, popt_quad_dh, quadratic_func)
        
        # Plot K_max fits
        ax_k.plot(x_fit, y_lin_k, '--', color='black', linewidth=2, alpha=0.8)
        ax_k.plot(x_fit, y_quad_k, '-', color=color, linewidth=2)
        
        # Plot dH_min fits
        ax_dh.plot(x_fit, y_lin_dh, '--', color='black', linewidth=2, alpha=0.8)
        ax_dh.plot(x_fit, y_quad_dh, '-', color=color, linewidth=2)
        
        # Add statistics to plots
        stats_text_k = f'Linear R² = {r2_lin_k:.3f}\nQuadratic R² = {r2_quad_k:.3f}\nImprovement = {r2_quad_k - r2_lin_k:+.3f}'
        stats_text_dh = f'Linear R² = {r2_lin_dh:.3f}\nQuadratic R² = {r2_quad_dh:.3f}\nImprovement = {r2_quad_dh - r2_lin_dh:+.3f}'
        
        ax_k.text(0.05, 0.15, stats_text_k, transform=ax_k.transAxes, fontsize=9, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax_dh.text(0.05, 0.15, stats_text_dh, transform=ax_dh.transAxes, fontsize=9, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        return (popt_lin_k, r2_lin_k, popt_quad_k, r2_quad_k), (popt_lin_dh, r2_lin_dh, popt_quad_dh, r2_quad_dh)
        
    except Exception as e:
        print(f"Error fitting for {station_name}: {e}")
        return None, None

# Create figures with 4 panels each
fig_k, axes_k = plt.subplots(2, 2, figsize=(15, 12))
fig_dh, axes_dh = plt.subplots(2, 2, figsize=(15, 12))

fig_k.suptitle(r'$K_{max}$ vs PPC Difference', fontsize=16)
fig_dh.suptitle(r'$\Delta H_{min}$ vs PPC Difference ', fontsize=16)

axes_k = axes_k.flatten()
axes_dh = axes_dh.flatten()

# Stations data
stations_data = [
    (ppc_lav_s2, 'LAV', 'blue', 0),
    (ppc_qro_s2, 'QRO', 'orange', 1),
    (ppc_rmy_s2, 'RMY', 'green', 2),
    (ppc_mzt_s2, 'MZT', 'purple', 3)
]

# Store fitted parameters
fitted_params = {}

# Create dispersion plots for each station
for ppc_data, station_name, color, idx in stations_data:
    k_fits, dh_fits = plot_station_dispersion(axes_k, axes_dh, ppc_data, K_max, dH_min, color, station_name, idx)
    fitted_params[station_name] = {'k_max': k_fits, 'dH_min': dh_fits}
    
    # Format K_max subplot
    axes_k[idx].set_xlabel('PPC Difference [%]')
    axes_k[idx].set_ylabel(r'$K_{max}$')
    axes_k[idx].grid(True, alpha=0.3)
    axes_k[idx].legend()
    axes_k[idx].set_ylim(0, 9)
    axes_k[idx].set_title(f'{station_name} ')
    
    # Format dH_min subplot
    axes_dh[idx].set_xlabel('PPC Difference [%]')
    axes_dh[idx].set_ylabel(r'$\Delta H_{min}$ [nT]')
    axes_dh[idx].grid(True, alpha=0.3)
    axes_dh[idx].legend()
    axes_dh[idx].set_title(f'{station_name} ')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Print fitted parameters
print("Linear and Quadratic Fit Parameters by Station:")
print("="*90)

for station_name in ['LAV', 'QRO', 'RMY', 'MZT']:
    fits = fitted_params.get(station_name, {})
    k_fits = fits.get('k_max')
    dh_fits = fits.get('dH_min')
    
    if k_fits is not None and dh_fits is not None:
        popt_lin_k, r2_lin_k, popt_quad_k, r2_quad_k = k_fits
        popt_lin_dh, r2_lin_dh, popt_quad_dh, r2_quad_dh = dh_fits
        
        print(f"\n{station_name}:")
        print("K_max:")
        print(f"  Linear:    K_max = {popt_lin_k[0]:.6f}·PPC + {popt_lin_k[1]:.3f}  (R² = {r2_lin_k:.3f})")
        print(f"  Quadratic: K_max = {popt_quad_k[0]:.6f}·PPC² + {popt_quad_k[1]:.4f}·PPC + {popt_quad_k[2]:.3f}  (R² = {r2_quad_k:.3f})")
        print(f"  Improvement: {r2_quad_k - r2_lin_k:+.3f}")
        
        print("dH_min:")
        print(f"  Linear:    dH_min = {popt_lin_dh[0]:.6f}·PPC + {popt_lin_dh[1]:.3f}  (R² = {r2_lin_dh:.3f})")
        print(f"  Quadratic: dH_min = {popt_quad_dh[0]:.6f}·PPC² + {popt_quad_dh[1]:.4f}·PPC + {popt_quad_dh[2]:.3f}  (R² = {r2_quad_dh:.3f})")
        print(f"  Improvement: {r2_quad_dh - r2_lin_dh:+.3f}")



sys.exit('end of child process')

mask_all = np.ones(len(df), dtype=bool)

mask_dH_75 = dH_min < -75
mask_dH_100 = dH_min < -100

mask_K_leq5 = K_max <= 5
mask_K_6_9 = (K_max >= 6) & (K_max <= 9)

# Define stations and their data (S2, S3, GPD)
stations = [
    ('LAV', s2_lav, s3_lav, gpd_lav),
    ('QRO', s2_qro, s3_qro, gpd_qro), 
    ('RMY', s2_rmy, s3_rmy, gpd_rmy),
    ('MZT', s2_mzt, s3_mzt, gpd_mzt)
]

# Define filtering conditions
filters = [
    ('All Cases', mask_all),
    ('dH_min < -75 nT', mask_dH_75),
    ('dH_min < -100 nT', mask_dH_100),
    ('K_max ≤ 5', mask_K_leq5),
    ('6 ≤ K_max ≤ 9', mask_K_6_9)
]

# Colors for each parameter
color_s2 = 'blue'
color_s3 = 'red' 
color_gpd = 'purple'

# Create separate figures for S2, S3, and GPD
parameters = [
    ('S2', [s2_lav, s2_qro, s2_rmy, s2_mzt], color_s2),
    ('S3', [s3_lav, s3_qro, s3_rmy, s3_mzt], color_s3),
    ('GPD', [gpd_lav, gpd_qro, gpd_rmy, gpd_mzt], color_gpd)
]

# Create individual figures for each parameter and filter condition
for param_name, param_data, param_color in parameters:
    for filter_name, filter_mask in filters:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'PDF Distribution of {param_name} - {filter_name}', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, (station_name, s2_data, s3_data, gpd_data) in enumerate(stations):
            ax = axes[idx]
            
            # Select the appropriate data for this parameter
            if param_name == 'S2':
                data = s2_data
            elif param_name == 'S3':
                data = s3_data
            else:  # GPD
                data = gpd_data
            
            # Apply filter and remove NaNs
            data_filtered = data[filter_mask & ~np.isnan(data)]
            
            if len(data_filtered) > 0:
                # Plot histogram
                bins = 80
                n, bins, patches = ax.hist(data_filtered, bins=bins, alpha=0.7, color=param_color, 
                                         density=True, edgecolor='black', linewidth=0.5)
                
                # Calculate statistics
                median = np.median(data_filtered)
                mean = np.mean(data_filtered)
                std = np.std(data_filtered)
                
                # Plot vertical lines
                ax.axvline(median, color='black', linestyle='-', linewidth=3, alpha=0.9)
                ax.axvline(mean - std, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax.axvline(mean + std, color='black', linestyle='--', linewidth=2, alpha=0.7)
                
                # Add statistics annotation
                textstr = f'n = {len(data_filtered)}\nMedian = {median:.2f}\nMean = {mean:.2f}\nσ = {std:.2f}'
                props = dict(boxstyle="round,pad=0.5", facecolor=param_color, alpha=0.2)
                ax.text(0.82, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
            
            ax.set_title(f'{station_name}', fontsize=12)
            ax.set_xlabel(f'{param_name} Value (nT)')
            ax.set_ylabel('Probability Density')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

# Print comprehensive summary statistics for each parameter separately
print("COMPREHENSIVE DISTRIBUTION STATISTICS SUMMARY")
print("=" * 90)

for param_name, param_data, param_color in parameters:
    print(f"\n{'='*50}")
    print(f"PARAMETER: {param_name}")
    print(f"{'='*50}")
    
    for filter_name, filter_mask in filters:
        print(f"\nFILTER: {filter_name}")
        print(f"{'-'*40}")
        print(f"{'Station':<8} {'Count':<6} {'Mean':<8} {'Median':<8} {'Std Dev':<8} {'Min':<8} {'Max':<8}")
        print(f"{'-'*70}")
        
        for idx, (station_name, s2_data, s3_data, gpd_data) in enumerate(stations):
            # Select the appropriate data for this parameter
            if param_name == 'S2':
                data = s2_data
            elif param_name == 'S3':
                data = s3_data
            else:  # GPD
                data = gpd_data
            
            # Apply filter and remove NaNs
            data_filtered = data[filter_mask & ~np.isnan(data)]
            
            if len(data_filtered) > 0:
                print(f"{station_name:<8} {len(data_filtered):<6} {np.mean(data_filtered):<8.2f} {np.median(data_filtered):<8.2f} "
                      f"{np.std(data_filtered):<8.2f} {np.min(data_filtered):<8.2f} {np.max(data_filtered):<8.2f}")
            else:
                print(f"{station_name:<8} {'0':<6} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<8}")

# Create summary tables for sample sizes
print("\n\nSAMPLE SIZE SUMMARY ACROSS ALL FILTERS AND STATIONS")
print("=" * 70)

for param_name, param_data, param_color in parameters:
    print(f"\n{param_name} SAMPLE SIZES:")
    print(f"{'Station':<8} {'All':<6} {'dH<-75':<6} {'dH<-100':<6} {'K≤5':<6} {'6≤K≤9':<6}")
    print(f"{'-'*70}")
    
    for idx, (station_name, s2_data, s3_data, gpd_data) in enumerate(stations):
        # Select the appropriate data for this parameter
        if param_name == 'S2':
            data = s2_data
        elif param_name == 'S3':
            data = s3_data
        else:  # GPD
            data = gpd_data
        
        counts = []
        for filter_name, filter_mask in filters:
            data_filtered = data[filter_mask & ~np.isnan(data)]
            counts.append(len(data_filtered))
        
        print(f"{station_name:<8} {counts[0]:<6} {counts[1]:<6} {counts[2]:<6} {counts[3]:<6} {counts[4]:<6}")