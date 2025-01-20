'''
En esta rutina tiene como propósito tomar un archivo de datos de índices 
geomagnéticos con una ventana de tiempo de varios días o anual, para 
fragmentar el archivo original, dividiendolo en varios archivos que abarcan un
día en sus respectivas ventanas de tiempo. Estos archivos pueden ser de tres 
tipos: dst, kp y sym
'''
################################################################################
#importación de librerías
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import re
import os
import sys
################################################################################ 
#introducimos la ventana de tiempo que abarca el archivo original y que está 
#indicada en el nombre del mismo
file_type = sys.argv[1]#=input("select file type option:\n dst: \n kp: \n sym: \n or \n ip: \n type:  ") 
#selección del tipo de archivo de acuerdo al índice geomagnético de interés

if file_type != 'ip':
    idate = input("format code:\n (yyyy-mm-dd): ")
    fdate = input("format code:\n (yyyy-mm-dd): ")
else:
     idate = input("format code:\n (yyyy-mm-dd): ")

code_stat = 'D' #código que indica el estado de los archivos, D, P o Q
                #(Definitivos, Provisionales o Rápidos)

path='/home/isaac/datos/'+file_type+'/'

code_name = 0   #código que indica el tipo de índice geomagnético en el nombre 
                #del archivo
                
head = 0        #número de líneas en el encabezado
sample = 0      #tiempo de muestreo

#revisión de si existe el archivo que se busca procesar, en función de la 
#ventana de tiempo y del tipo de índice de interés seleccionado. 
if file_type == 'dst':
    if os.path.isfile(path+'Dst_'+idate+'_'+fdate+'_D'+'.dat'):
        code_stat = '_D'
        file_type = 'dst'
        code_name = 'Dst_'  
        head    = 22                
        
    elif os.path.isfile(path+'Dst_'+idate+'_'+fdate+'_P'+'.dat'):
        code_stat = '_P'
        file_type = 'dst'
        code_name = 'Dst_'
        head    = 22   
         
    else:
        code_stat = '_Q'
        file_type = 'dst'
        code_name = 'Dst_'
        head    = 22     
     
elif file_type == 'kp':        
    if os.path.isfile(path+'Kp_'+idate+'_'+fdate+'_D'+'.dat'):
        code_stat = '_D'
        file_type = 'kp'
        code_name = 'Kp_'  
        head    = 35      
         
    elif os.path.isfile(path+'Kp_'+idate+'_'+fdate+'_P'+'.dat'):
        code_stat = '_P'
        file_type = 'kp'
        code_name = 'Kp_'
        head    = 35         
    else:
        code_stat = '_Q'
        file_type = 'kp'
        code_name = 'Kp_' 
        head    = 35

elif file_type == 'ip':        
    if os.path.isfile(path+idate+'.dat'):
        file_type = 'ip'           
        
elif file_type == 'sym':
    sample=input('chose sample rate: \n h or m: ')
    
    if os.path.isfile(path+'ASY_'+idate+'_'+fdate+'h_D'+'.dat'):
        if sample=='h':    
            code_stat = 'h_D'
        else:
            code_stat = 'm_D'
                        
        file_type = 'sym'
        code_name = 'ASY'
                
    if os.path.isfile(path+'ASY_'+idate+'_'+fdate+'h_P'+'.dat'):
        if sample=='h':    
            code_stat = 'h_P'
        else:
            code_stat = 'm_P'
        file_type = 'sym'
        code_name = 'ASY'
        head    = 24
        
    #else:
       # if sample=='h':    
      #      code_stat = 'h_Q'
     #   else:
     #       code_stat = 'm_Q'    
    #    file_type = 'sym'
   #     code_name = 'ASY'
  #      head    = 24 
#                         print(idx1)
else:
    print('file does not exist.\n Try again with another date or another name')        
################################################################################
################################################################################
################################################################################   
#generación del DataFrame
if  file_type == 'kp':
    df = pd.read_csv(path+code_name+idate+'_'+fdate+code_stat+'.dat', header=head, sep='\s+')
    df = df.drop(columns=['|'])

elif file_type == 'dst':
    df = pd.read_csv(path+code_name+idate+'_'+fdate+code_stat+'.dat', header=head, sep='\s+')
    df = df.drop(columns=['|'])
        
elif file_type == 'ip':
    print(path+idate+'.dat')
    df = pd.read_csv(path+idate+'.dat', header=None, sep='\s+')    

elif file_type == 'sym':#sym-H

    df = pd.read_csv(path+'ASY_'+idate+'_'+fdate+'m_P.dat', header=head, sep='\s+')
    df = df.drop(columns=['|'])                                          

################################################################################
################################################################################
#fragmentación del archivo original en varios archivos con un día de ventana de
#tiempo
################################################################################
################################################################################

enddata = 0 #hora del del día en que debe terminar la ventana de tiempo

idx = 0     #vector de tiempo a generar

step = 0    #indica la tasa de muestreo, dependiendo del índice geomagnético 
            #seleccionado
            
if file_type == 'dst':
    enddata = fdate+ ' 23:00:00'
    idx = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='H')
    step=24
    fhour=23                    
elif file_type == 'kp':   
    enddata = fdate+ ' 21:00:00'                     
    idx = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='3H')
    step=8
    fhour=7
    
elif file_type == 'sym':
    if sample =='h':
        enddata = fdate+ ' 23:00:00'
        idx = pd.date_range(start = pd.Timestamp(idate), \
                            end = pd.Timestamp(enddata), freq='H')
        step=24
        fhour=23                                 
    else:
        enddata = fdate+ ' 23:59:00'
        idx = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='T')
        step=1440
        fhour=1439

elif file_type == 'ip':
    year = df.iloc[:,0]
    doy  = df.iloc[:,1]   
    #la primer columna es el año, la segunda es el DOY. 
    df['combined'] = df.iloc[:, 0]*1000 + df.iloc[:,1] 
    #obtenemos una columna para la fecha
    df["dt"] = pd.to_datetime(df["combined"], format = "%Y%j") 
    dt = df.iloc[:, 58]
    idate = dt[0]
    fdate = dt[len(dt)-1]
    fdate = str(fdate)
    fdate = fdate[0:10]  
    enddata = fdate+ ' 23:00:00'      
    idx = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='H')
    step=24
    fhour=23                                                     
else:
    print('try with another sample rate')      

df = df.set_index(idx)

if file_type != 'ip':
    df = df.drop(columns=['DATE', 'TIME'])

df = df.reset_index()    
df.rename(columns=df.iloc[0]).drop(df.index[0])
               
for i in range(0,len(idx),step):
    mask = (df['index'] >= idx[i]) & (df['index'] <= idx[i+fhour])
    df_new = df[mask]
   # sym_H  = df_new['SYM-H'].apply(lambda x: '{0:0>4}'.format(x))
    date = str(idx[i])
    date = date[0:10]
    
    if file_type == 'dst' or file_type == 'kp':
        name_new = file_type+'_'+date+'.dat'
        new_path = '/home/isaac/datos/'+\
        file_type+'/daily/'
        df_new = df_new.drop(columns=['DOY'])
        df_new.to_csv(new_path+name_new, sep= '\t', index=False)  
        #np.savetxt(new_path+name_new, df_new, fmt=['%4i %2i %2i %2i %2i %2i %3i %+4i'])
    elif file_type == 'ip':
        df_new = df.drop(columns=['combined', 'dt'])
        df_new = df_new.reset_index()
        df_new = df_new.drop(columns=['index'])
        name_new = file_type+'_'+date+'.dat'
        new_path = '/home/isaac/datos/'+\
        file_type+'/daily/'
        df_new = df_new.T        
        df_new.to_csv(new_path+name_new, sep= '', index=False)
                
    else:
        if sample== 'm':
            df_new = df_new.drop(columns=['DOY'])
            df_new.rename(columns={'level_0':'index'})
            df_new = df_new.drop(columns=['index'])            
            name_new = file_type+'_'+date+code_stat+'.dat'
            new_path = '/home/isaac/datos/'+\
            file_type+'/daily/'
            #print(df_new)
          #  df_new.to_csv(new_path+name_new, sep= '\t', index=False)        
            with open(new_path+name_new, 'w') as f:
                for index, row in df_new.iterrows():
                    formatted_line = f"{int(row['ASY-D']):>4} {int(row['ASY-H']):>4} {int(row['SYM-D']):>5} {int(row['SYM-H']):>5} \n"
                    f.write(formatted_line)    
        else:
            df_new = df_new.drop(columns=['ASY-D', 'ASY-D']) 
            df_new = df_new.drop(columns=['index'])       
            name_new = file_type+'_'+date+code_stat+'h.dat'
            new_path = '/home/isaac/datos/'+file_type+'/daily/'
            
          #  sym_H.to_csv(new_path+name_new, sep= '\t', index=False)             








