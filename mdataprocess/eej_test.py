import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from aux_time_DF import index_gen, convert_date
from scipy.interpolate import interp1d

from magnetic_datstruct import get_dataframe
from magdata_processing import base_line, get_diurnalvar
from scipy.fft import fft, fftfreq
from scipy import signal

st= sys.argv[1]
idate = sys.argv[2]# "formato(yyyymmdd)"
fdate = sys.argv[3]

enddata = fdate+ ' 23:59:00'
idx = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='T')
idx_hr = pd.date_range(start = pd.Timestamp(idate), \
                    end = pd.Timestamp(enddata), freq='H')    
idx_daily = pd.date_range(start = pd.Timestamp(idate), \
                        end = pd.Timestamp(enddata), freq='D')

filenames = []
dates = []
for i in idx_daily:
    date_name = str(i)[0:10]
    dates.append(date_name)
    date_name_newf = convert_date(date_name,'%Y-%m-%d', '%Y%m%d')
    new_name = str(date_name_newf)[2:8]
    fname = st+'_'+new_name+'.min'
    filenames.append(fname)

# Function to calculate the number of filter coefficients
def n_terms(wdiff, A, n):
    M = round((42 / (2.285 * wdiff)) + 1)
    N_terms = (M * 2) + 1
    if N_terms > n:
        M_over = N_terms - n
        N_terms2 = N_terms - M_over
        M = (N_terms2 - 1) // 2
    return M

def fill_gap(data):

    def nan_helper(y):    
        return np.isnan(y), lambda z: z.nonzero()[0]   
    
    nans, non_nans_indices = nan_helper(data)    
    
    valid_data_indices = non_nans_indices(~nans)
    valid_data_values = data[valid_data_indices]
    
    cubic_interp = interp1d(valid_data_indices, valid_data_values, kind='cubic', fill_value="extrapolate")
    
    data[nans] = cubic_interp(non_nans_indices(nans))
    
    return data
###############################################################################}
path = '/home/isaac/MEGAsync/datos/jicamarca/'+st+'/'
path_qdl = '/home/isaac/tools/test/test_isaac/' 

data = get_dataframe(filenames, path, idx, dates)
inicio = data.index[0]
final =  data.index[-1]

#monthly base line
baseline_line, baseline_curve, DD = base_line(data, idx, idx_daily)

H_det = data-baseline_line

diurnal_baseline, offset = get_diurnalvar(H_det, idx_daily, st)

H_r = H_det - offset
H_n = H_r - diurnal_baseline

H = np.array(H_r)
H = fill_gap(H)

H_d = np.array(H_n)
H_d = fill_gap(H_d)
#print(np.isnan(H).sum())


list_of_days = ['2024-01-13', '2024-01-14', '2024-01-15','2024-01-16', '2024-01-17', '2024-01-18']

#plt.plot(data.index, H)
#plt.plot(data.index, diurnal_baseline)
#plt.show()

dt = 60

h = 3600            

f_ny = f_ny = 1.0 / (2.0 * dt)  # Here, dt is in minutes, so 1 minute -> 1/60 Hz

n = len(H)

time = np.linspace(0, (n *dt)-1, n)+1  # Time axis in minutes

fk = time/(n*dt)

H = np.array(H)

F_H = fft(H)


f, PWD_H = signal.periodogram(H, 1/60)

plt.figure(figsize=(8, 6), dpi=200)
plt.title('Power Spectrum Density, huan st.')
plt.semilogy(f, PWD_H)
plt.xscale('log')
plt.ylim(np.min(10e-5), np.max(PWD_H))
plt.savefig("/home/isaac/tools/test/test_isaac/"+\
            st+'_PSD_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()

f = 1/(48*h)
fr = f * ((2 * np.pi) * h)
M = n_terms(fr, 50, n)


Pxx, freqs, times, cax = plt.specgram(H, Fs=1/60, noverlap=1, NFFT=M, scale='dB',\
                   cmap='plasma')
plt.title('Espectrograma, huan')
axes_spec = plt.gca()
plt.xlabel("Time (s)")
plt.ylabel("Frequency (hz)")
plt.colorbar(cax, label='dH Hz⁻¹').ax.yaxis.set_label_position('left')
plt.tight_layout()
plt.savefig("/home/isaac/tools/test/test_isaac/"+\
            st+'_spec_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()



f, PWD_H = signal.periodogram(H_d, 1/60)

plt.figure(figsize=(8, 6), dpi=200)
plt.title('Power Spectrum Density, huan st. no diurnal variation')
plt.semilogy(f, PWD_H)
plt.xscale('log')
plt.ylim(np.min(10e-5), np.max(PWD_H))
plt.savefig("/home/isaac/tools/test/test_isaac/"+\
            st+'_PWD_noSQ_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()

Pxx, freqs, times, cax = plt.specgram(H_d, Fs=1/60, noverlap=1, NFFT=M, scale='dB',\
                   cmap='plasma')
plt.title('Espectrograma, huan')
axes_spec = plt.gca()
plt.xlabel("Time (s)")
plt.ylabel("Frequency (hz)")

plt.colorbar(cax, label='dH Hz⁻¹').ax.yaxis.set_label_position('left')
plt.tight_layout()
plt.savefig("/home/isaac/tools/test/test_isaac/"+\
            st+'_spec_noSQ_'+str(inicio)[0:10]+"_"+str(final)[0:10]+".png")
plt.show()


'''

#fk = fftfreq(n, dt/60 )  # Frequency axis in Hz (1/minute converted to Hz)
print(fk)

#



'''