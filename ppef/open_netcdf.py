import xarray as xr 
import numpy as np 
import sys
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from scipy import integrate
#import matplotlib.path as mpath
#from matplotlib.colors import ListedColormap
import os
import glob
import aacgmv2
module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)
from modules.obs_info import obs_info

path = '/home/isaac/datos/ampere/'

date = sys.argv[1]

ihour = sys.argv[2] # float number
iminute = sys.argv[3]

tmp = f"{path}{date}.*.86400.600.north.grd.ncdf"
filename = glob.glob(tmp)

#filename = f'{path}{date}.0600.86400.600.north.grd.ncdf'



ds = xr.open_dataset(filename[0])

def prep_j(j_par, nlon, nlat, jrmin, jrmax):

    jr2d = np.reshape(j_par, ( nlon.item(), nlat.item(),))
    jr2d = jr2d.T
    
    jr2d = np.flip(jr2d, axis=1)

    jr2d = np.vstack([jr2d, jr2d[0, :]])
    
    from scipy.ndimage import zoom

    target_ny = nlat.item() * 10
    target_nx = nlon.item() * 10

    zoom_y = target_ny / jr2d.shape[0]
    zoom_x = target_nx / jr2d.shape[1]

    jr2d = zoom(jr2d, (zoom_y, zoom_x), order=1)
    jr2d_cleaned = np.where(np.abs(jr2d) <= 0.2, 0, jr2d)
    
    jr2d_scaled = ((jr2d - jrmin) / (jrmax - jrmin) * 255).astype(int)
    jr2d_scaled = np.clip(jr2d_scaled, 0, 255)
    
    
    return(jr2d_cleaned)

def obs_mlon(obs):
    
    data = []
    
    for i in obs:
        #print(f'Observatorio: {i.lower()}')
        if i.lower() == 'teo':
            net = 'regmex'
        else:
            net = 'intermagnet'
        
        info = obs_info(net, i.lower())
        mlon = float(info[9])     # station magnetic longitude
        hemi = info[10]

        if hemi == 'W':
            mlon = -mlon 
        #print(i.lower(), mlon)
        data.append(mlon)
    return(data)



def obs_mlt(obs_mlon, dt):
    mlt_data = []
    for i in range(len(obs_mlon)):
        mlt = aacgmv2.convert_mlt(obs_mlon[i], dt, m2a=False)
        mlt_data.append(mlt.item())   
    
    return(mlt_data)

def int_superficie(j_rad, theta_colat,lat1,lat2):
    #theta_array
    
    theta2 = (lat2*np.pi)/180
    theta1 = (lat1*np.pi)/180  
    theta_rad = (theta_colat * np.pi) / 180
    mask = (theta_colat >= lat1) & (theta_colat <= lat2)
    theta_range = theta_rad[mask]
    j_range = j_rad[mask]
    
    mask2 = np.abs(j_range) >= 0.2
    
    j_filtered = j_range[mask2]
    theta_filtered = theta_range[mask2]

    rad = 7151000 #altura de la orbita de medicion  

    I_net = (rad**2) * integrate.simpson(   y=j_range * np.sin(theta_range),  x=theta_range)
    
    return I_net

def plot_j(J_mag, latmin, lonmin, latmax, lonmax, MLT_obs, formatted_date, date_name):
    # --- Datos base ---
    mlts = list(MLT_obs.values()) 
    obs = list(MLT_obs.keys())
    
    ny, nx = J_mag.shape

    # Bordes en lat/lon
    lat_edges = np.linspace(latmin, latmax, ny+1)
    lon_edges = np.linspace(lonmax, lonmin, nx+1)

    # Centros para graficar
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

    # --- Conversión a coordenadas polares ---
    r = lat2d - latmin
    
    theta = np.deg2rad(lon2d)

    # --- Gráfico polar ---
    fig = plt.figure(figsize=(10,10))
    #ax = plt.subplot(111, polar=True)
    ax = fig.add_axes([0.18, 0.18, 0.67, 0.67], polar=True)
    # Mapa de densidad
    c = ax.pcolormesh(theta, r, J_mag,
                    cmap='seismic', vmin=jrmin, vmax=jrmax, shading='auto')

    print(f'J max: {jrmax}, J min: {jrmin}')
    # Ajustes estéticos para orientación MLT
    ax.set_theta_zero_location("S")   # 0 MLT abajo
    ax.set_theta_direction(1)         # sentido antihorario: 12 arriba, 6 derecha, 18 izquierda
    

   # r_zoom_min = latmax - lat_zoom_max # inner radius (closer to pole)
    r_min = 0,
    r_max = int(latmax-latmin)

    #ax.set_rlim(r_max, 0)
    ax.set_rticks(np.arange(0, r_max+1, 10))
    ax.set_theta_offset(np.pi/2)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    
    # Etiquetas de MLT
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])) 
    #ax.set_xticklabels(["12 MLT", "18 MLT", "0 MLT", "6 MLT"]) # inverted order


    sectors = [0, 6, 12, 18]
    for i in range(4):
        angle = np.deg2rad((sectors[i]*15) - 180)
        ax.text(angle, 52, f"{sectors[i]} MLT", fontsize = 15, color ="black",
                    ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8))
    
    

    print('Obs \t J net [M A]')
    for h in range(len(mlts)):
        mlon_obs = mlts[h] * 15 - 180
        lon_idx = int(np.argmin(np.abs(lon_centers - mlon_obs)))
        j_rad = J_mag[:, lon_idx]
        j_net = int_superficie(j_rad, r[:,0], 5, 40)

        print(f'{obs[h]} {(j_net/1e12):.2f}')  
              
        
        angle = np.deg2rad(mlon_obs)    
        ax.text(angle, 48, f"$I_{{net}}$({obs[h]}) \n {(j_net/1e12):.2f} MA", fontsize = 18, color ="darkgreen",
                    ha="center", va="center")
        ax.plot(angle,38,marker='o',markersize=10,
                color='darkgreen',markeredgecolor='black',zorder=5)
##################################################################################################3
           # tu cálculo de longitud en grados


        
    # Manual colorbar axes: [left, bottom, width, height]
    cb = fig.add_axes([0.89, 0.07, 0.03, 0.30])
    cb = plt.colorbar(c, orientation="vertical", cax=cb,  pad=0.2) 
    cb.set_label(r"$\mu A/m^2$", fontsize=20)
    cb.ax.tick_params(labelsize=16)

    fminute = int(iminute) + 10
    fig.text(0.86, 0.98, f'{formatted_date} \n {int(ihour):02d}:{int(iminute):02d} - {int(ihour):02d}:{fminute:02d} UT', 
            ha='center', va='top', fontsize=20, 
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8))
    
    plt.savefig(f'/home/isaac/longitudinal_studio/fig/ampere/{date_name}.{ihour}.{iminute}.png', dpi=300)
    plt.close()

    return


if sys.argv[3] == "help":
    print("=== ATTRIBUTES ===")
    print(ds.attrs)

    print("\n=== DIMENSIONS ===")
    for dim, size in ds.dims.items():
        print(f"{dim}: {size}")

    print("\n=== VARIABLES ===")
    for var in ds.variables:
        print(f"{var}: {ds[var].shape}")


else:
    
    # set limit variables
    dimensions = [1000, 500]

    latmin = 50.0
    latmax = 90.0
    lonmin = -180
    lonmax = 180

    dlat = 10
    dlon = 90
    dlatmin = 40

    arrow_scl = 2000.    
    
    #target time
    time = ds.time.values
    avgint = ds.avgint.values

    target_time = float(ihour) + (float(iminute) / 60.0)
    
 
    idx = np.where(np.abs(time - target_time) <= (avgint / 3600.0) / 2.0)[0]
    
    iyear = ds.year.values[idx]
    idoy = ds.doy.values[idx]
    itime = time[idx]
    iavrs = avgint[idx]

    radius = ds.R
    #print(f"Forma: {radius.shape}")
    #print(f"Mínimo: {radius.min().values:.2f} km")
    #print(f"Máximo: {radius.max().values:.2f} km")
    #print(f"Promedio: {radius.mean().values:.2f} km")
    #sys.exit('end')
    dt = datetime(iyear.item(), 1, 1, int(ihour), int(iminute), tzinfo=timezone.utc) + timedelta(days=idoy.item() - 1)

    formatted_date = dt.strftime("%Y/%m/%d")
    date_name = dt.strftime("%Y%m%d")
    #grid info
    nlat = ds.nLatGrid.values[idx]
    nlon = ds.nLonGrid.values[idx]
    
    colat = ds.cLat_deg.values[idx,:]
    geocolat = ds.geo_cLat_deg.values[idx,:]

    mlt = ds.mlt_hr.values[idx,:]    
    lat = 90.0 - colat
    lon = mlt * 15.0
    
    #vectors
    dbnorth1 = ds.db_T.values[idx,:]    
    dbeast1 = ds.db_P.values[idx,:]    
    dbnorth2 = ds.db_Ph_Th.values[idx,:]    
    dbeast2 = ds.db_Ph_Ph.values[idx,:]    
    
    dens_curr = ds['jPar']
    j_dim = dens_curr.dims
    j_par = dens_curr.values[idx, :]
    
    jrmin = -2#np.max(j_par)
    jrmax = 2#np.min(j_par)
    obs = ["TEO", "SJG", "GUI", "TAM", "JAI", "BMT", "KAK", "HON"]
    
    mlon_data = obs_mlon(obs)
    obs_mlt = obs_mlt(mlon_data, dt)
    
    mlt_dict = dict(zip(obs, obs_mlt))
    
    jr2d = prep_j(j_par, nlon, nlat,jrmin, jrmax)
 

 
    plot_j = plot_j(jr2d, latmin, lonmin, latmax, lonmax, mlt_dict, formatted_date, date_name)





    
    
    
    
    
    
    
    
    
    
    
    
