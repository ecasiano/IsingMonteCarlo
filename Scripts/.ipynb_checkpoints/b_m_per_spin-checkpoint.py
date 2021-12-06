# Evaluate average energy per spin
import numpy as np
import matplotlib.pyplot as plt
import os

def m_exact_func(T):
    K = 1/T
    return (1-(1-np.tanh(K)**2)**4/(16*np.tanh(K)**4))**(1/8)

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
for subdir, dirs, files in os.walk('../../Data_reduced/'):
    for filename in files:
        if filename.split('_')[1] == str(L): L_filenames.append(filename)
L_filenames = sorted(L_filenames)

# Compute energy per spin for all temperatures in Data directory
m = []
throw_bins=128
std_errors_all_T = []
for file in L_filenames:
    data = np.loadtxt('../../Data_reduced/'+file)
    
    # COMPUTE ERROR BARS #
    #Determine max bin level
    max_bin_level = int(np.log2(np.shape(data)[0]))
    min_bin = 32

    #Initialize list to save standard error
    std_errors = []
    
    #Binning loop
    binned_data = np.copy(data[:,1][throw_bins:])
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
    
    m_mean = np.mean(data[:,1][throw_bins:]/total_sites)
    m.append(m_mean)
m = np.array(m)
std_errors_all_T = np.array(std_errors_all_T)

# Retrieve temperatures for which we have data and store to array
temperatures = []
for filename in L_filenames:
    temperatures.append(float(filename.split('_')[3].replace('.dat','')))
T = np.array(temperatures)

# ---- INCLUDE THEORETICAL FORMULA -------#
T_exact = np.linspace(T[0],T[-1],1000)
m_exact = np.nan_to_num(m_exact_func(T_exact))

# COMPUTE ERROR BARS #

# Plot magnetization per spin vs temperature
# plt.plot(T,m,'o',label=r'$L=16$',color='darkorange')
plt.errorbar(T,m, yerr=std_errors_all_T, fmt='.', capsize=5,label=r"$L=16$",marker='s',zorder=1,alpha=0.50,color='darkorange');
plt.plot(T_exact,m_exact,'-',label=r'$exact$',zorder=0,color='bisque')
plt.ylabel(r'$\langle m \rangle$')
plt.xlabel("T");
plt.legend(frameon=False)
plt.savefig("../Figures/b_m_per_spin.svg",dpi=400)