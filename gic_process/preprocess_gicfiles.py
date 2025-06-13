import os
import re
import csv
from pathlib import Path
from datetime import datetime
import sys

year_dir = sys.argv[1]# "formato(yyyymmdd)"
station = sys.argv[2]
csv_file = sys.argv[3]
print(f'file input name: {csv_file}')
output_file = f"{csv_file}.dat"#"Datos_GICS_2022-05-27 RMY.csv.dat"
#sys.argv[4]
# Configuration
#station_dir = f'/home/isaac/datos/gics_obs/{year_dir}/{station}/'   # Update this path
output_suffix = ".dat"
encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

#csv_file = f'{station_dir}{csv_file}'

def process_file(csv_file, output_file):
    for encoding in encodings_to_try:
        try:
            with open(csv_file, 'r', encoding=encoding) as infile:
                first_line = infile.readline()
                infile.seek(0)
                
                with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                    writer = csv.writer(outfile)
                    reader = csv.reader(infile)
                    
                    for parts in reader:
                        if not parts:
                            continue
                            
                        date_part = parts[0].strip()
                       # if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", date_part): 
                       #     print(date_part)
                        processed = False
                        # Case 1: ISO format with T/Z (2025-05-12T12:00:00Z)
                        
                        if re.match(r'^[2-9]\d{3}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', date_part):
                            try:
                                dt = datetime.strptime(date_part, "%Y-%m-%dT%H:%M:%SZ")
                                parts[0] = dt.strftime("%Y-%m-%d %H:%M:%S")
                                writer.writerow(parts)
                                processed = True
                            except ValueError:
                                pass
                        
                        # Case 2: Quoted ISO format ("2025-05-12")
                        if not processed and re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", date_part): 
                            try:
                                cleaned = date_part.strip('"')
                                dt = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
                                parts[0] = dt.strftime("%Y-%m-%d %H:%M:%S")
                                                
                                writer.writerow(parts)
                                processed = True
                            except ValueError:
                                pass
                        
                        # Case 3: Quoted or unquoted dd/mm/yyyy HH:MM ("12/05/2025 12:00")
                        if not processed:
                            # Remove quotes if present
                            datetime_str = date_part.strip('"')
                            
                            # Try with seconds first, then without
                            try:
                                dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
                                parts[0] = dt.strftime("%Y-%m-%d %H:%M:%S")
                                writer.writerow(parts)
                                processed = True
                            except ValueError:
                                try:
                                    dt = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M")
                                    parts[0] = dt.strftime("%Y-%m-%d %H:%M:%S")
                                    writer.writerow(parts)
                                    processed = True
                                except ValueError:
                                    pass
                        
                        
                        if not processed:
                            print(f"Skipping unprocessable format: {date_part}")
                
                return True
              
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            return False

    print(f"Failed to decode {csv_file} with tried encodings")
    return False
# Process each CSV file

output_file = f"{csv_file}{output_suffix}"
if process_file(csv_file, output_file):
    print(f"Successfully processed: {output_file}")