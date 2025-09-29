import shelve 
import os
import pandas

os.chdir("/home/isaac/datos/gics_obs/")
with shelve.open("gicdata2024.dat", "r") as shelf:
    gicdata2024 = shelf["gicdata2024"]