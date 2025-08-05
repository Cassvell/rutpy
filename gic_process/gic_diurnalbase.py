from gic_threshold import threshold
import pandas as pd
import os
import sys 
# Get the absolute path to the directory containing your module
module_dir = os.path.abspath('/home/isaac/rutpy/mdataprocess') 
sys.path.append(module_dir)

# Now you can import the module
import magdata_processing 


from magdata_processing import get_diurnalvar, get_qd_dd
from night_time import night_time



def gic_diurnalbase(gic, stat):   
       
    stddev_20 = gic.resample('20Min').std().fillna(method='ffill')
    threshold_value, indices = threshold(stddev_20)
    exceed_indices = stddev_20[stddev_20 > threshold_value].index
    
    print(f'Exceeding indices for {stat}: {len(exceed_indices)}')
    idx_daily = pd.date_range(start = gic.index[719], end = gic.index[-722], freq= 'D' )
    qd, offset = get_diurnalvar(gic[719:-721], idx_daily, 'regmex', stat.lower())
    
    return(gic[719:-721] - qd, qd)