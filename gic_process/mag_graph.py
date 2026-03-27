import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gicdproc import df_dH, df_Kloc
import sys
from datetime import datetime, timedelta
H_stat = sys.argv[1]
i_date = sys.argv[2]

# Set f_date - use provided value or default to i_date
f_date = sys.argv[3] 

dir_path = f'/home/isaac/datos/dH_{str(H_stat)}/'

fdate = datetime.strptime(f_date, '%Y%m%d')
fdate2 = fdate + timedelta(days=1)
fdate2 = str(fdate2.strftime('%Y%m%d'))

H = df_dH(i_date, f_date, dir_path, H_stat)

###############################################################################
###############################################################################
dir_path = '/home/isaac/datos/Kmex/'
k = df_Kloc(i_date, f_date, dir_path, H_stat)
k = round(k)

colorsValue = []
for i, value in enumerate(k):
    if value > 9:
        k[i] = np.nan  # Set value to NaN if it's greater than 9
        colorsValue.append('gray')  # Optional: You can assign a color for NaN values
    elif value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')

k_index = np.argmax(k)
H = np.argmin(H)

print(k)
sys.exit()
k = k/10

colorsValue = []

for value in k['K']:  
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')
 
        

# Define time limits
inicio_k = pd.Timestamp('2026-03-20 00:00:00')
final_k = pd.Timestamp('2026-03-26 21:00:00')
fig, ax = plt.subplots(2, figsize=(12, 14))

pos = ax[0].get_position()
ax[0].set_position([pos.x0, pos.y0* 0.9, pos.width, pos.height])

ax[0].bar(k.index, k['K'], width=0.1, align='edge', color=colorsValue)
ax[0].set_ylim(0, 9)
ax[0].set_title(r'Estimated regional $K_{mex}$ index', fontsize=20)
ax[0].set_ylabel(r'$K_{mex}$', fontweight='bold', fontsize=20)
ax[0].set_xlabel(r'UTC [Begin: 2024/05/10 00:00 UTC]', fontweight='bold', fontsize=20)
ax[0].set_xlim(inicio_k, final_k)
ax[0].grid(color='gray', linestyle='--', linewidth=0.5)
ax[0].xaxis.set_major_locator(mdates.DayLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d'))

# Create a twin Axes sharing the x-axis
ax2 = ax[0].twinx()
y_ticks = [5, 6, 7, 8, 9]
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(['G1', 'G2', 'G3', 'G4', 'G5'])

# Custom legend for K index
legend_elements = [
    Line2D([0], [0], color='green', lw=4, label='Quiet'),
    Line2D([0], [0], color='yellow', lw=4, label='Disturb'),
    Line2D([0], [0], color='red', lw=4, label='Storm')
]
ax[0].legend(
    handles=legend_elements,
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.3),
    ncol=3
)



# First plot: DH values
inicio_H = pd.Timestamp('2026-03-20 01:00:00')
final_H = pd.Timestamp('2026-03-26 23:00:00')

pos = ax[1].get_position()
ax[1].set_position([pos.x0, pos.y0* 0.3, pos.width, pos.height])

ax[1].plot(H, color='k')
ax[1].set_title('Estimated regional hourly-averaged variations of '+r'$B_H$', fontsize=20)
ax[1].set_ylabel(r'$\Delta H_{mex}$ [nT]', fontweight='bold', fontsize=20)
ax[1].set_xlabel(r'UTC [Begin: 2024/05/10 00:00 UTC]', fontweight='bold', fontsize=20)

# Add horizontal lines with labels
ax[1].axhline(y=-250, color='red', linestyle='--', label='Intense')
ax[1].axhline(y=-100, color='yellow', linestyle='--', label='Moderate')
ax[1].axhline(y=-50, color='green', linestyle='--', label='Weak')

# Set limits and grid for H
ax[1].set_xlim(inicio_H, final_H)
ax[1].grid(color='gray', linestyle='--', linewidth=0.5)
ax[1].xaxis.set_major_locator(mdates.DayLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d'))

# Legend for H plot
ax[1].legend(loc='lower center', 
    bbox_to_anchor=(0.5, -0.3),
    ncol=3
)
textstr2 = 'REGMEX/LANCE (http://regmex.unam.mx)'
textstr = r'$K_{mex}/ \Delta H_{mex}$: Regional early values of K/H values for Mexico by'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)  # Adjust alpha for visibility
ax[1].text(0.05, 1.2, textstr2, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='bottom')
ax[1].text(0.05, 1.3, textstr, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='bottom')

plt.tight_layout()  # Optional: to adjust spacing
plt.savefig(path+'idx.png')







