import pandas as pd
import shelve
import dbm
import os

path = '/home/isaac/datos/gics_obs/2024/'
base_filename = f'{path}gicdata2024.dat.db'

# 1. Check what files actually exist
files = [f for f in os.listdir(path) if 'gicdata2024' in f]
print("Found files:", files)

# 2. Try different filename variations
possible_filenames = [
    base_filename,  # No extension
    f'{base_filename}.dat',
    f'{base_filename}.db',
]

for filename in possible_filenames:
    print(f"\nTrying: {filename}")
    print(f"File exists: {os.path.isfile(filename)}")
    
    if os.path.isfile(filename):
        try:
            # Try with shelve
            with shelve.open(filename, 'r') as shelf:
                data = shelf['gicdata2024']
            print("✓ Success with shelve")
            break
        except Exception as e:
            print(f"✗ Shelve failed: {e}")
            
        try:
            # Try with dbm directly
            with dbm.open(filename, 'r') as db:
                print("✓ Can open with dbm directly")
        except Exception as e:
            print(f"✗ dbm also failed: {e}")