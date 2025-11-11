import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from gicdproc import gic_qd


idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

stat  = ['LAV', 'QRO', 'RMY','MZT']
dir_path = f'/home/isaac/datos/gics_obs/qdl/'

qd_lav = gic_qd(idate, fdate, dir_path, 'LAV')
qd_qro = gic_qd(idate, fdate, dir_path, 'QRO')
qd_rmy = gic_qd(idate, fdate, dir_path, 'RMY')
qd_mzt = gic_qd(idate, fdate, dir_path, 'MZT')


fig, axes = plt.subplots(4, 1, figsize=(12, 16))  # 4 rows, 1 column

# Panel 1: LAV
axes[0].plot(qd_lav.iloc[:,0], color='black', linewidth=4, label='LAV Base')
axes[0].plot(qd_lav.iloc[:,0] + qd_lav.iloc[:,1], color='red', alpha=0.7, label='LAV ± Uncertainty')
axes[0].plot(qd_lav.iloc[:,0] - qd_lav.iloc[:,1], color='red', alpha=0.7)
axes[0].set_title('LAV - QD Model')
axes[0].set_ylabel('GIC')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2: QRO
axes[1].plot(qd_qro.iloc[:,0], color='black', linewidth=4, label='QRO Base')
axes[1].plot(qd_qro.iloc[:,0] + qd_qro.iloc[:,1], color='red', alpha=0.7, label='QRO ± Uncertainty')
axes[1].plot(qd_qro.iloc[:,0] - qd_qro.iloc[:,1], color='red', alpha=0.7)
axes[1].set_title('QRO - QD Model')
axes[1].set_ylabel('GIC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Panel 3: RMY
axes[2].plot(qd_rmy.iloc[:,0], color='black', linewidth=4, label='RMY Base')
axes[2].plot(qd_rmy.iloc[:,0] + qd_rmy.iloc[:,1], color='red', alpha=0.7, label='RMY ± Uncertainty')
axes[2].plot(qd_rmy.iloc[:,0] - qd_rmy.iloc[:,1], color='red', alpha=0.7)
axes[2].set_title('RMY - QD Model')
axes[2].set_ylabel('GIC')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Panel 4: MZT
axes[3].plot(qd_mzt.iloc[:,0], color='black', linewidth=4, label='MZT Base')
axes[3].plot(qd_mzt.iloc[:,0] + qd_mzt.iloc[:,1], color='red', alpha=0.7, label='MZT ± Uncertainty')
axes[3].plot(qd_mzt.iloc[:,0] - qd_mzt.iloc[:,1], color='red', alpha=0.7)
axes[3].set_title('MZT - QD Model')
axes[3].set_ylabel('GIC')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/isaac/QD_gics.png', dpi=300)
plt.show()

#for i in range(len(stat)):
#    c