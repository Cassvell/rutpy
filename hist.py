import ftplib
import ftputil
import fileinput
from scipy import stats
from scipy import fftpack
from scipy import signal
from ts_acc import fixer, mz_score, despike, dejump
from gicdproc import pproc
import sys
import os

import glob
#import gzip
import pandas as pd
import numpy as np
year_dir = 2023# "formato(yyyymmdd)"
stat = 'QRO'

data_dir='/home/isaac/MEGAsync/datos/gics_obs/'+str(year_dir)+'/'+stat+'/'
list_fnames = sorted(glob.glob(data_dir + "/*.dat"))
lastfile = list_fnames[-4:]
#print(last2files)
print(lastfile)
