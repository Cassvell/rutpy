
import re
import datetime
import pandas as pd
# Define the regular expression for an irregular space delimiter
delimiter_pattern = re.compile(r'\s+')
directory = '/home/isaac/datos/ip/Ey/'
cols = {"year" : [], "doy" : [], "hour" : [], "minute" : [], "Bt" : [], "Bx" : [], "By" : [], "Bz" : [], "Vx" : [], "Vy" : [], "Vz" : []\
    , "n_p" : [], "T_p" : [], "Pdyn" : [], "E": [], "beta" : []}

with open(directory + 'omni_1min_20152.dat', 'r') as f:
    # Read and process each line manually
    for line in f:
        # Split the line using the regular expression delimiter
        row = re.split(delimiter_pattern, line.strip())
        
        
        for  i, key in enumerate(cols.keys()):
           # print(key)
            (cols[key]).append(row[i])
        #print(cols["Bz"]) 
    
for key in cols:
    try:
        if key in ["year", "doy", "hour", "minute"]:
            # Convert to int for these keys
            cols[key] = [int(x) for x in cols[key]]
        else:
            # Convert to float for these keys
            cols[key] = [float(x) for x in cols[key]]
    except ValueError:
        print(f"Conversion error for key {key} with values: {cols[key]}")

# Print the converted data

idx = pd.date_range(start = pd.Timestamp('2015-03-01'), end = pd.Timestamp('2015-3-31'), freq='D')
dates = idx.strftime('%Y%m%d')
step=1440

data = pd.DataFrame.from_dict(cols)
''''''
for i in range(int(len(data) / step)):
    #print(data[(i*288):((i+1)*288)-1])
    daily_data = data[i * step:(i + 1) * step]

  #  print(len(daily_data))
    new_path = '/home/isaac/datos/ip/Ey/daily/'
    fname = "ip_"+dates[i]+'1min.dat'
    # Open the file and write formatted data
    with open(new_path + fname, 'w') as f:
        for index, row in daily_data.iterrows():
            formatted_line = f"{int(row['year']):>4} {int(row['doy']):>4} {int(row['hour']):>3} {int(row['minute']):>3}{row['Bt']:8.2f} {row['Bx']:8.2f} {row['By']:8.2f} {row['Bz']:8.2f} {row['Vx']:8.1f} {row['Vy']:8.1f} {row['Vz']:8.1f} {row['n_p']:7.2f} {row['T_p']:9.0f} {row['Pdyn']:6.2f} {row['E']:7.2f} {row['beta']:8.1f}\n"
            f.write(formatted_line)
