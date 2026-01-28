import xarray as xr 
import numpy as np 
import sys
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.path as mpath
from matplotlib.colors import ListedColormap
import aacgmv2
path = '/home/isaac/datos/ampere/'
filename = f'{path}20150317.ncdf'

ds = xr.open_dataset(filename)

ihour = sys.argv[1] # float number
iminute = sys.argv[2]

def prep_j(j_par, nlon, nlat, jrmin, jrmax):
    
    jr2d = np.reshape(j_par, (nlat.item(), nlon.item()))
    
    jr2d = jr2d.T
    
    jr2d = np.flip(jr2d, axis=1)

    jr2d = np.vstack([jr2d, jr2d[0, :]])
    
    from scipy.ndimage import zoom

    target_ny = nlat.item() * 10
    target_nx = nlon.item() * 10

    zoom_y = target_ny / jr2d.shape[0]
    zoom_x = target_nx / jr2d.shape[1]

    jr2d = zoom(jr2d, (zoom_y, zoom_x), order=1)
    
    jr2d_scaled = ((jr2d - jrmin) / (jrmax - jrmin) * 255).astype(int)
    jr2d_scaled = np.clip(jr2d_scaled, 0, 255)
    
    
    return(jr2d)

def plot_j(jr, jr2d, latmin, lonmin, latmax, lonmax, nlat, nlon, dt):
    
    J_mag = np.ma.masked_where(np.abs(jr2d) <= 0.2, jr2d)
    fig = plt.figure(figsize=(17, 17))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))


    ax.set_extent([lonmin, lonmax, latmin, latmax],crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center
    circle = mpath.Path(verts)

    # Set the boundary
    ax.set_boundary(circle, transform=ax.transAxes)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, xlocs=np.arange(-180, 181, 90),
        ylocs=np.arange(60, 91, 20),linewidth=0.6,color="gray",alpha=0.9)

    globe = plt.imread("globe_aacgm_800km.jpg")
    mlt_hours = np.arange(0, 24) # 0–23 hours
    
    mlon = mlt_hours * 15.0
    
    shift_px = int(mlon[0] * globe.shape[1] / 360.0)
    globe = np.roll(globe, -shift_px, axis=1)
    
    #ax.imshow(globe,origin="upper",transform=ccrs.PlateCarree(),extent=[-180, 180, -90, 90],alpha=0.9,zorder=0)     
   
    ny, nx = jr2d.shape
    lat_edges = np.linspace(latmin, latmax, ny + 1)
    lon_edges = np.linspace(lonmin, lonmax, nx + 1)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)    
    # Crear malla de puntos
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    # Dibujar puntos sobre el eje polar
    #ax.scatter(lon2d, lat2d,
    #        transform=ccrs.PlateCarree(),
    #        s=3, color="black", alpha=0.6, zorder=10)

    img = ax.pcolormesh(lon_edges,lat_edges,J_mag,transform=ccrs.PlateCarree(),cmap='seismic',vmin=jrmin,
                        vmax=jrmax,alpha=0.8)
    


    # Add MLT labels like a clock
    mlt_labels = {0: (0, latmin-5), 6: (90, latmin-5), 12: (180, latmin-5), 18: (-90, latmin-5)}
    for hour, (lon, lat) in mlt_labels.items():
        ax.text(lon, lat, f"{hour:02d} MLT",
                transform=ccrs.PlateCarree(),
                ha="center", va="center", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8), zorder=5)

    
    cax = fig.add_axes([0.90, 0.70, 0.02, 0.15])

    cb = plt.colorbar(img,cax=cax,orientation="vertical")

    cb.set_label(r"$\mu A/m^2$")    
   
    #plt.title("Projected Vector Field (Polar)")

    plt.show()


if sys.argv[1] == "help":
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

    latmin = 60.0
    latmax = 90.0
    lonmin = -180
    lonmax = 180

    dlat = 10
    dlon = 90
    dlatmin = 40

    jrmin = -2.0
    jrmax = 2.0

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

    
    dt = datetime(iyear.item(), 1, 1, int(ihour), int(iminute), tzinfo=timezone.utc) + timedelta(days=idoy.item() - 1)
    #grid info
    nlat = ds.nLatGrid.values[idx]
    nlon = ds.nLonGrid.values[idx]
    
    colat = ds.cLat_deg.values[idx]
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

    
    jr2d = prep_j(j_par, nlon, nlat,jrmin, jrmax)
 
    
    plot_j = plot_j(j_par, jr2d, latmin, lonmin, latmax, lonmax, nlat, nlon, dt)
    
    #print(lat)





    
    
    
    
    
    
    
    
    
    
    
    
