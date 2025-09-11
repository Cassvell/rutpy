import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

idate = sys.argv[1]
fdate = sys.argv[2]

path = '/home/isaac/datos/pca/'
obs = ['teo', 'gui', 'jai', 'kak']

#col0 = asymh
#col1 = diono
#col2 = ppef

tw = 0

days = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(fdate), freq='D')
idx = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(fdate), freq='min')

for st in obs:
    df = pd.read_csv(f'{path}_{st}_{str(idate)}')