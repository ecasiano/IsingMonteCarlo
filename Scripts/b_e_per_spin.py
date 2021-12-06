# Evaluate average energy per spin
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import ellipk

def e_exact_func(T):
    K = 1/T
    q = 2*np.sinh(2*K)/np.cosh(2*K)**2
    K1 = ellipk(q*q)
    return -(1/np.tanh(2*K))*(1+(2/np.pi)*(2*np.tanh(2*K)**2-1)*K1)

def get_std_error(mc_data):
    '''Input array and calculate standard error'''
    N_bins = np.shape(mc_data)[0]
    std_error = np.std(mc_data)/np.sqrt(N_bins)
    
    return std_error

def get_binned_data(mc_data):
    '''Return neighbor averaged data.'''
    N_bins = np.shape(mc_data)[0]
    start_bin = N_bins % 2
    binned_mc_data = 0.5*(mc_data[start_bin::2]+mc_data[start_bin+1::2]) #Averages (A0,A1), (A2,A3), + ... A0 ignored if odd data

    return binned_mc_data

def get_autocorrelation_time(error_data):
    '''Given an array of standard errors, calculates autocorrelation time'''
    print(error_data[0],error_data[-2])
    autocorr_time = 0.5*((error_data[-2]/error_data[0])**2 - 1)
    return autocorr_time

# --- Main --- #

# Set system size
L = 16
total_sites = L**2

# Load desired L files
L_filenames = []
for subdir, dirs, files in os.walk('./Data_reduced/'):
    for filename in files:
        if filename.split('_')[1] == str(L): L_filenames.append(filename)
L_filenames = sorted(L_filenames)

# Compute energy per spin for all temperatures in Data directory
e = []
throw_bins=128
std_errors_all_T = []
for file in L_filenames:
    data = np.loadtxt('./Data_reduced/'+file)
    
    # COMPUTE ERROR BARS #
    #Determine max bin level
    max_bin_level = int(np.log2(np.shape(data)[0]))
    min_bin = 32

    #Initialize list to save standard error
    std_errors = []
    
    #Binning loop
    binned_data = np.copy(data[:,0][throw_bins:])
    for i in range(max_bin_level):
#         print(np.shape(binned_data)[0])
        std_errors.append(get_std_error(binned_data))   
        if np.shape(binned_data)[0]/2 < min_bin: break
        binned_data = get_binned_data(binned_data)
    std_errors = np.array(std_errors)
    
    # Error propagation
    std_errors /= total_sites
    
    # Save error for this T
    std_errors_all_T.append(np.max(std_errors))
    
    e_mean = np.mean(data[:,0][throw_bins:]/total_sites)
    e.append(e_mean)
e = np.array(e)
std_errors_all_T = np.array(std_errors_all_T)

# Retrieve temperatures for which we have data and store to array
temperatures = []
for filename in L_filenames:
    temperatures.append(float(filename.split('_')[3].replace('.dat','')))
T = np.array(temperatures)

# ---- INCLUDE THEORETICAL FORMULA -------#
T_exact = np.linspace(T[0],T[-1],1000)
e_exact = e_exact_func(T_exact)
    
# Plot energy per spin vs temperature
plt.errorbar(T,e, yerr=std_errors_all_T, fmt='.', capsize=5,label=r"$L=16$",marker='s',zorder=1,alpha=0.50);
plt.plot(T_exact,e_exact,'-',label=r'$exact$',zorder=0,color='lightblue')
plt.ylabel(r'$\langle e \rangle$')
plt.xlabel("T");
plt.legend(frameon=False)
plt.savefig("Figures/b_e_per_spin.pdf",dpi=400)