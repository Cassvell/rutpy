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
path = '/home/isaac/geomstorm/rutidl' 

#prueba de funcion
#net = 'intermagnet'
#obs = 'sjg'

def night_time(net, obs):
    obs_info = []
    with open(f"{path}/{net}_stations.csv") as f:
        data = csv.reader(f)
        for row in data:
            
            if obs == row[2]:
                obs_info = row

    return obs_info