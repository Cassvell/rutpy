import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



path = '/home/isaac/datos/ppef_model/'
common_name = 'PPFM_'

obs = ['gui', 'jai', 'kak', 'teo']


df_teo = pd.read_csv(f'{path}{common_name}{obs[3]}.csv')
date = df_teo.iloc[:,0]
date = pd.to_datetime(date)
df_teo = df_teo.set_axis(date)
ppef_teo = df_teo['Penetration (mV/m)']
B = np.diff(ppef_teo)

# Create figure with 2 subplots (vertically stacked)
fig, ax = plt.subplots(2, 1, figsize=(8, 6))  # Corrected figsize spelling

# Plot data on each subplot
ax[0].plot(ppef_teo)  # Use ax[0] for first subplot (Python uses 0-based indexing)
ax[0].set_title('PPEF Teo')
ax[0].set_ylabel('Value')

ax[1].plot(date[1:], B)  # Use ax[1] for second subplot
ax[1].set_title('B Component')
ax[1].set_ylabel('Value')
ax[1].set_xlabel('Time/Index')  # Add x-label to bottom plot

plt.tight_layout()  # Adjusts spacing between subplots
plt.show()