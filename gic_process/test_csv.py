import csv
import pandas as pd
from collections import defaultdict
path = '/home/isaac/datos/gics_obs/2025/LAV/daily/'
filename = 'GIC_2025-05-14_LAV (Copiar).dat'
filename2 = 'GIC_2025-05-13_LAV.dat'

#df1 = pd.read_csv(path+filename2, sep=r'\s+')

#df2 = pd.read_csv(path+filename, sep=r'\s+')
try:
    # Read first 5 rows
    df = pd.read_csv(path + filename, sep='\t')
    
    # Count empty values
    empty_counts = df.isna().sum()  # Count NaN values
    empty_strings = (df == '').sum()  # Count empty strings
    total_empty = empty_counts + empty_strings
    
    # Check if any empty values exist (sum across all columns)
    if total_empty.sum() > 0:
        print(f"Found {total_empty.sum()} empty values to replace")
        
        # Replace both NaN and empty strings
        df.replace({'': -999.999}, inplace=True)
        df.fillna(-999.999, inplace=True)
        
        # Save back to same file (consider using a different filename for safety)
        output_filename = filename
        df.to_csv(path + output_filename, sep='\t', index=False, na_rep='-999.999')
        print(f"File saved with empty values replaced: {path + output_filename}")
    else:
        print("No empty values found in first 5 rows")
        
except FileNotFoundError:
    print(f"Error: File not found - {path + filename}")
except Exception as e:
    print(f"Error occurred: {str(e)}")