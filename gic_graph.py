#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:11:27 2023

@author: isaac
"""

import matplotlib.pyplot as plt
from gicdproc import pproc, reproc, df_dH, df_Kloc
from timeit import default_timer as timer
import sys
start = timer()
year_dir = str(idate[0:4])
"""
date_name = input("write initial date in format yyyy-mm-dd \n >  " )

node = ['LAV', 'MZT', 'RMY', 'QRO']

def node_dataframe(node, date):
    dir_path = '/home/isaac/MEGAsync/datos/gics_obs/2023/'

    df = pd.read_csv(dir_path+'Datos_GICS_'+date+' '+node+'.csv', header=0, sep=',',\
                 skip_blank_lines=True, encoding='latin1')    
    df['DateTime']  = pd.to_datetime(df.iloc[:,0], format='%Y-%m-%d %H:%M:%S')   
    df = df.set_index(df['DateTime']) 
    df = df.drop(columns=['DateTime'])
    
    gic = df.iloc[:,0]
    
    gic = gic.replace('NO DATA', np.NaN)
    return(gic)
#gic = gic.astype(str)

#rank = np.arange(float(min(gic)), 5, float(max(gic)))

"""

df_qro = pproc('QRO', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/QRO/')
df_lav = pproc('LAV', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/LAV/')
df_rmy = pproc('RMY', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'/RMY/')
df_mzt = pproc('MZT', data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+year_dir+'//MZT/')


#print(df_lav['LAV'].gic[0:])

idate = sys.argv[1]# "formato(yyyymmdd)"
fdate = sys.argv[2]

#idate = input("write initial date in format yyyy-mm-dd \n >  " )
#fdate = input("write final date in format yyyy-mm-dd \n >  " )    
gicTW_lav = (df_lav['LAV'].gic_proc[idate:fdate])
gicTW_qro = (df_qro['QRO'].gic_proc[idate:fdate])
gicTW_rmy = (df_rmy['RMY'].gic_proc[idate:fdate])
gicTW_mzt = (df_mzt['MZT'].gic_proc[idate:fdate])


T1TW_lav = (df_lav['LAV'].T1_proc[idate:fdate])
T1TW_qro = (df_qro['QRO'].T1_proc[idate:fdate])
T1TW_rmy = (df_rmy['RMY'].T1_proc[idate:fdate])
T1TW_mzt = (df_mzt['MZT'].T1_proc[idate:fdate])

T2TW_lav = (df_lav['LAV'].T2_proc[idate:fdate])
T2TW_qro = (df_qro['QRO'].T2_proc[idate:fdate])
T2TW_rmy = (df_rmy['RMY'].T2_proc[idate:fdate])
T2TW_mzt = (df_mzt['MZT'].T2_proc[idate:fdate])
###############################################################################
###############################################################################
dir_path = '/home/isaac/MEGAsync/datos/dH_teo/'
H = df_dH(idate, fdate, dir_path)
###############################################################################
###############################################################################
dir_path = '/home/isaac/MEGAsync/datos/Kmex/'
k = df_Kloc(idate, fdate, dir_path)
k = round(k)

colorsValue = []
for value in k:
    if value < 4:
        colorsValue.append('green')
    elif value == 4:
        colorsValue.append('yellow')
    else:
        colorsValue.append('red')



inicio = gicTW_lav.index[0]
final  = gicTW_lav.index[-1]
##############################################################################################
#fig 1
##############################################################################################
fig, ax = plt.subplots(6, figsize=(12,14))
fig.suptitle('Estudio de GICs, 2023', fontsize=24, fontweight='bold')

ax[0].plot(gicTW_lav)
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_title('LAV st', fontsize=18)
ax[0].set_ylabel(' GIC [A]', fontweight='bold')

ax[1].plot(gicTW_qro)
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_title('QRO st', fontsize=18)
ax[1].set_ylabel(' GIC [A]', fontweight='bold')

ax[2].plot(gicTW_rmy)
ax[2].grid()
ax[2].set_xlim(inicio,final)
ax[2].set_title('RMY st', fontsize=18)
ax[2].set_ylabel(' GIC [A]', fontweight='bold')


ax[3].plot(gicTW_mzt)
ax[3].grid()
ax[3].set_xlim(inicio,final)
ax[3].set_title('MZT st', fontsize=18)
ax[3].set_ylabel(' GIC [A]', fontweight='bold')

ax[4].plot(H, color='k')
ax[4].set_ylabel(' DH [nT]', fontweight='bold')
ax[4].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[4].grid()
ax[4].set_xlim(inicio,final)


ax[5].bar(k.index, k, width = 0.1, align='edge', color=colorsValue)
ax[5].set_ylim(0,9)
ax[5].set_xlim(inicio,final)
ax[5].set_ylabel(' Kcoe', fontweight='bold')
ax[5].grid()

fig.tight_layout()

fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir+"/gic_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()
##############################################################################################
#fig 2
##############################################################################################
fig, ax = plt.subplots(6, figsize=(12,14))
fig.suptitle('Estudio de GICs, 2023', fontsize=24, fontweight='bold')

ax[0].plot(T1TW_lav, label='T1')
ax[0].plot(T2TW_lav, label='T2')
ax[0].grid()
ax[0].set_xlim(inicio,final)
ax[0].set_title('LAV st', fontsize=18)
ax[0].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[0].legend()

ax[1].plot(T1TW_qro, label='T1')
ax[1].plot(T2TW_qro, label='T2')
ax[1].grid()
ax[1].set_xlim(inicio,final)
ax[1].set_title('QRO st', fontsize=18)
ax[1].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[1].legend()

ax[2].plot(T1TW_rmy, label='T1')
ax[2].plot(T2TW_rmy, label='T2')
ax[2].grid()
ax[2].set_xlim(inicio,final)
ax[2].set_title('RMY st', fontsize=18)
ax[2].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[2].legend()

ax[3].plot(T1TW_mzt, label='T1')
ax[3].plot(T2TW_mzt, label='T2')
ax[3].grid()
ax[3].set_xlim(inicio,final)
ax[3].set_title('MZT st', fontsize=18)
ax[3].set_ylabel(' Temperatura [C°]', fontweight='bold')
ax[3].legend()

ax[4].plot(H, color='k')
ax[4].set_ylabel(' DH [nT]', fontweight='bold')
ax[4].set_title('Indices geomagnéticos, Estación Coeneo', fontsize=18)
ax[4].grid()
ax[4].set_xlim(inicio,final)


ax[5].bar(k.index, k, width = 0.1, align='edge', color=colorsValue)
ax[5].set_ylim(0,9)
ax[5].set_xlim(inicio,final)
ax[5].set_ylabel(' Kcoe', fontweight='bold')
ax[5].grid()

fig.tight_layout()

fig.savefig("/home/isaac/geomstorm/rutpy/gicsOutput/"+year_dir+"/T_obs_"+\
            str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()
end = timer()

print(end - start)
'''

'''
