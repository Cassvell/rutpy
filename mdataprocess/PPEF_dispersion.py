import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from threshold import max_IQR
from scipy.stats import genpareto, kstest #anderson
import kneed as kn
from lmoments3 import distr


#st= sys.argv[1]
idate = sys.argv[1]# "formato(yyyy-mm-dd)"
fdate = sys.argv[2]
path2 = '/home/isaac/longitudinal_studio/fig/ppef_dist/'
st = ['lzh', 'bmt', 'tam', 'jai', 'cyg', 'teo', 'hon', 'gui',  'kak', 'sjg']
color = ['red', 'olive', 'seagreen', 'magenta', 'purple', 'orange', 'darkgreen', 'salmon', 'sienna', 'gray']
for i in range(len(st)):
    print(f'Observatorio: {st[i]}')
    path = '/home/isaac/datos/pca/'

    df = pd.read_csv(f'{path}{st[i]}_{idate}_{fdate}.dat', sep='\s+')
    
    dp2 = df.iloc[:, 3]
    nbins = int(len(dp2) / 60)
    dp2 = np.abs(dp2)
    dp2 = np.array(dp2)  

    dp2 = dp2[~np.isnan(dp2)]
    dp2 = np.unique(dp2)

    # Get histogram data
    frequencies, bin_edges = np.histogram(dp2, bins=nbins * 2, density=True)

    # Plot the histogram for visualization

    stddev = np.std(dp2)
    #print(f'std: {stddev}')
    #print(f'bins: {bin_edges}')
    
    plt.hist(dp2, density=True, bins=nbins * 2, color='navy',histtype='stepfilled', alpha=0.6)
    plt.axvline(x=stddev*2, linestyle='--', color=color[i], label=f'{st[i].upper()}: {stddev*2:.2f} nT')    
    #plt.axvline(x=threshold, color='k', linestyle='-', label=f'Threshold: {threshold2:.2f}')
    plt.xlim(0, 70)
    plt.title(r"$\mathrm{H_{PPEF}}$ Threshold")
    plt.xlabel(r"$\mathrm{|H_{PPEF}|}$ distribution [nT]")
    plt.ylabel('probability')
    plt.legend()
    plt.savefig(f'{path2}{st[i]}_{idate}_{fdate}.png')
