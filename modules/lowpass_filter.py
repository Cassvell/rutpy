import numpy as np
import cmath
from scipy import signal

def aphase(array):
    
    """ Calcula la fase geometrica (en radianes) de todos los elementos de 
    un array 1D complejo"""
    
    output=[]
    for j in range(len(array)):
        pp = cmath.phase(array[j])
        output.append(pp)
        
    output = np.array(output)
    return output 

def dcomb(n, m, f, freqs):
    
    """ Crea un peine de Dirac de dimensions [n,m], para un conjunto de 
    frecuencias predeterminadas en un vector o lista de frecuencias dadas """
    
    delta = np.zeros([n, m])    
    for w in freqs:    
        idx = np.where(abs(f) == w)
        uimp = signal.unit_impulse([n,m], idx)
        delta = delta + uimp 
    
    return delta