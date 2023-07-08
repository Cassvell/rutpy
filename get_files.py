# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## para ejecutar es python Plot_V1.py  arch1.m arch2.m yyyy-mm-dd HH:MM:SS
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import paramiko

from PIL import Image
from datetime import datetime, timedelta
import glob, os
import fnmatch 

import paramiko

idate = input("write initial date in format yyyymmdd \n >  " )
fdate = input("write final date in format yyyymmdd \n >  " )   

remote_path= '/data/output/indexes/coeneo/'
localfilepath  = '/home/isaac/MEGAsync/datos/dH_coe/'

def get_files(initial_date, final_date, remote_path, localfilepath, select_fnames):
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='132.247.181.60', username='carlos', password='c4rl0s1s44c')
    
    
    sftp = ssh.open_sftp()
    sftp.chdir(remote_path)
    
    
    files = [] #lista de los archivos, ordenada del último al primero
    for entry in sorted(sftp.listdir_attr(),\
                        key=lambda k: k.st_mtime, reverse=True):
        files.append(entry.filename)   
    
    
    for entry in select_fnames:
        sftp.get(entry, localfilepath+entry) 
    
    return()

idx_daylist = pd.date_range(start = pd.Timestamp(idate), \
                                      end = pd.Timestamp(fdate), freq='D')
idx_list = (idx_daylist.strftime('%Y%m%d')) 

str1 = "coe_"
ext = ".delta_H.early"
def list_names(daterange, string1, string2):
    select_fnames = []
    for i in idx_list:
        tmp_name = string1+str(i)+string2
        select_fnames.append(tmp_name)
    return(select_fnames)


list_fnames = list_names(idx_list, str1, ext)

wget = get_files(idate, fdate, remote_path, localfilepath, list_fnames)

"""

"""

       
