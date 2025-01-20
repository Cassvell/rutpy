
import re
import datetime
import pandas as pd
# Define the regular expression for an irregular space delimiter
delimiter_pattern = re.compile(r'\s+')
directory = '/home/isaac/datos/ip/Ey/'
cols = {"year" : [], "doy" : [], "hour" : [], "minute" : [], "Bz" : [], "Vx" : [], "n_p" : [], "T_p" : [], "Pdyn" : [], "E": []}

with open(directory + 'omni_5min_2015.dat', 'r') as f:
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

idx = pd.date_range(start = pd.Timestamp('2015-01-01'), end = pd.Timestamp('2015-12-31'), freq='D')
dates = idx.strftime('%Y%m%d')
step=288

data = pd.DataFrame.from_dict(cols)
''''''
for i in range(int(len(data) / step)):
    #print(data[(i*288):((i+1)*288)-1])
    daily_data = data[i * step:(i + 1) * step]

  #  print(len(daily_data))
    new_path = '/home/isaac/datos/ip/Ey/daily/'
    fname = "ip_"+dates[i]+'.dat'
    # Open the file and write formatted data
    with open(new_path + fname, 'w') as f:
        for index, row in daily_data.iterrows():
            formatted_line = f"{int(row['year']):>4} {int(row['doy']):>4} {int(row['hour']):>3} {int(row['minute']):>3} {row['Bz']:8.2f} {row['Vx']:8.1f} {row['n_p']:7.2f} {row['T_p']:9.0f} {row['Pdyn']:6.2f} {row['E']:7.2f}\n"
            f.write(formatted_line)
