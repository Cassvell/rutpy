import xarray as xr 
import numpy as np 


path = '/home/isaac/datos/ampere/'
filename = f'{path}20150307.0500.86400.600.north.grd.ncdf'


ds = xr.open_dataset(filename)
        
# Print all metadata
print("=== ATTRIBUTES ===")
print(ds.attrs)
print("\n=== DIMENSIONS ===")
for dim, size in ds.dims.items():
    print(f"{dim}: {size}")
print("\n=== VARIABLES ===")
for var in ds.variables:
    print(f"{var}: {ds[var].shape}")
