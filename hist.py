import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path='/home/isaac/Escritorio/proyecto/master_thesis/datos/'

df = pd.read_csv(path+'corr_tab.dat', sep='\s+', header=None)



df.replace(99.999999, np.nan, inplace=True)

corr1 = df.iloc[:,0]
corr2 = df.iloc[:,1]

plt.hist(corr2)
plt.show()
