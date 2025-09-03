#function 
#obs_info:
#col 0: indice
#col 1: observatorio
#col 2: codigo iaga obs
#col 3: latitud geográfica
#col 4: hemisferio N o S
#col 5: longitud magnética
#col 6: hemisferio E  ó W
#col 7: latitud magnética
#col 8: hemisferio N ó S
#col 9: longitud magnética
#col 10: hemisferio E ó W
#col 11: Hora UTC

import csv
#<<<<<<< Updated upstream

#=======
import os
#>>>>>>> Stashed changes

#if not os.path.exists(path) or not os.path.isdir(path):

#    path = '/home/isaac/rutidl'
#    if not os.path.exists(path) or not os.path.isdir(path):
#        raise FileNotFoundError(f"Directory not found: {path}")

#print(f"{path} directory exists.")

# Function to get observation info




def night_time(net, obs):
    obs_info = []
    path = '/home/isaac/datos' 
    file_path = f"{path}/{net}_stations.csv"
    
    #info_tl = mlt()
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the CSV file
    with open(file_path, 'r') as f:
        data = csv.reader(f)
        for row in data:
            if obs == row[2]:  # Assuming the observation code is in the 3rd column
                obs_info = row
                #print(f"Observation info found: {obs_info}")
                break  # Exit loop once the observation is found
    
    if not obs_info:
        raise ValueError(f"Observation '{obs}' not found in {file_path}")
    
    return obs_info