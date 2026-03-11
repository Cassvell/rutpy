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
import pandas as pd 
import aacgmv2
import datetime
import time
#>>>>>>> Stashed changes

#if not os.path.exists(path) or not os.path.isdir(path):

#    path = '/home/isaac/rutidl'
#    if not os.path.exists(path) or not os.path.isdir(path):
#        raise FileNotFoundError(f"Directory not found: {path}")

#print(f"{path} directory exists.")

# Function to get observation info




def obs_info(net, obs):
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




def obs_mlon(obs):
    
    data = []
    regmex_obs = ['teo', 'coe', 'itu', 'lav', 'mzt', 'qro', 'rmy']
    if isinstance(obs, list):
        for i in obs:
            #print(f'Observatorio: {i.lower()}')
            if i.lower() == 'teo' or i.lower() == 'coe':
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
            
    if isinstance(obs, str):
    
        if obs in regmex_obs:
            net = 'regmex'
        else:
            net = 'intermagnet'
            
        info = obs_info(net, obs)
        mlon = float(info[9])     # station magnetic longitude
        hemi = info[10]
        
        if hemi == 'W':
            mlon = -mlon 
            
        data = mlon
    return(data)

def obs_mlt(mlon, dt):
    mlt = aacgmv2.convert_mlt(mlon, dt, m2a=False)
    return(round(mlt[0]))

def compute_mlt_ts(mlon, time_m, time_d):
    
    """
    Returns MLT in hours (0–24) for a station over time_m
    """
    mlt_edges = []
    
    index=[0,-1]
    for j in range(2):
        tmp_mlt = aacgmv2.convert_mlt(mlon, time_m[j], m2a=False)  
        #print(obs_mlon)      
        duration = datetime.timedelta(hours=tmp_mlt.item())
        total_minutes_td = duration.total_seconds() / 60
        seconds = total_minutes_td * 60
        time_format = time.strftime("%H:%M:%S", time.gmtime(seconds))
        mlt_edges.append(time_format)
    
    idate2 = time_d[0] 
    fdate2 = time_d[-1] + datetime.timedelta(days=2)
    
    if datetime.datetime.strptime(mlt_edges[0], "%H:%M:%S").time() >= datetime.time(12, 0):
        idate2 = time_d[0] - datetime.timedelta(days=2)
        fdate2 = fdate2 - datetime.timedelta(days=2)      
    else:
        idate2 = idate2
        fdate2 = fdate2

    start_date_str = f"{idate2.year}-{idate2.month:02d}-{idate2.day:02d} {mlt_edges[0]}"
    end_date_str = f"{fdate2.year}-{fdate2.month:02d}-{fdate2.day:02d} {mlt_edges[-1]}"
    # Crear el date_range

    mlt_series = pd.date_range(start=start_date_str, end=end_date_str, freq='min')   
    
    ndays = int(len(time_m)/1440)
    return mlt_series[0:ndays*1440]