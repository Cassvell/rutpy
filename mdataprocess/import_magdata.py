import pandas as pd
import numpy as np
#from statistics import mode
#from datetime import datetime
# Ajuste de distribuciones
import sys
#from numpy.linalg import LinAlgError
#from scipy.interpolate import splrep, splev
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter1d
#from scipy.interpolate import NearestNDInterpolator
from magnetic_datstruct import get_dataframe
from magdata_processing import base_line, get_diurnalvar

idate = sys.argv[1]
fdate = sys.argv[2]

mag = sys.argv[3]

path = '///'