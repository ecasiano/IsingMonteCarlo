# Evaluate average energy per spin
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import ellipe,ellipk

def Cv_exact_func(T):
    K = 1/T
    q = 2*np.sinh(2*K)/np.cosh(2*K)**2
    K1 = ellipk(q*q)
    E1 = ellipe(q*q)
    B = (1-np.tanh(2*K)**2)*(np.pi/2 + (2*np.tanh(2*K)**2-1)*K1)
    return (4/np.pi)*(K/np.tanh(2*K))**2*(K1-E1-B)

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
for L in [16,20,24][::-1]:
    # Set system size
    # L = 16
    total_sites = L**2

    # Load desired L files
    L_filenames = []
    for subdir, dirs, files in os.walk('./Data/'):
        for filename in files:
            if filename.split('_')[1] == str(L): L_filenames.append(filename)
    L_filenames = sorted(L_filenames)

    # Compute energy for all temperatures in Data directory
    E = []
    throw_bins=128
    std_errors_all_T_E = []
    for file in L_filenames:
        data = np.loadtxt('./Data/'+file)
        
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

        # Save error for this T
        std_errors_all_T_E.append(np.max(std_errors))

        E_mean = np.mean(data[:,0][throw_bins:])
        E.append(E_mean)
    E = np.array(E)
    std_errors_all_T_E = np.array(std_errors_all_T_E)

    # Same as above but for e squared
    E2 = []
    throw_bins=128
    std_errors_all_T_E2 = []
    for file in L_filenames:

        data = np.loadtxt('./Data/'+file)
        
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

        # Save error for this T
        std_errors_all_T_E2.append(np.max(std_errors))

        E2_mean = np.mean(data[:,2][throw_bins:])
        E2.append(E2_mean)
    E2 = np.array(E2)
    std_errors_all_T_E2 = np.array(std_errors_all_T_E2)

    # Retrieve temperatures for which we have data and store to array
    temperatures = []
    for filename in L_filenames:
        temperatures.append(float(filename.split('_')[3].replace('.dat','')))
    T = np.array(temperatures)

    # Compute specific heat for all temperatures
    Cv = (E2-E**2)/T**2
    Cv /= total_sites
    
    # Error propgation
    std_errors_all_T_E /= (T**2*total_sites)
    std_errors_all_T_E2 /= (T**2*total_sites)
    std_errors_all_T_E *= (2*std_errors_all_T_E*E_mean)
    std_errors_all_T = np.sqrt(std_errors_all_T_E**2+std_errors_all_T_E**2)
    
    # ---- INCLUDE THEORETICAL FORMULA -------#
    T_exact = np.linspace(T[0],T[-1],1000)
    Cv_exact = Cv_exact_func(T_exact)

    # COMPUTE ERROR BARS #

    # Plot specific heat per spin vs temperature
#     plt.plot(T,Cv,'o',label=r'$L=%d$'%L)
    plt.errorbar(T,Cv, yerr=std_errors_all_T, fmt='.', capsize=5,label=r'$L=%d,T_{\rm{max}}\approx %.2f$'%(L,T[np.argmax(Cv)]),marker='s',zorder=1,alpha=0.50);
    if L==16:
        plt.plot(T_exact,Cv_exact,'-',label=r'$exact,T_{\rm{max}}\approx %.2f$'%(T_exact[np.argmax(Cv_exact)]),zorder=0)
    plt.ylabel(r'$\langle c_V \rangle$')
    plt.xlabel("T");
    plt.legend(frameon=False)
plt.savefig("Figures/c_specific_heat.pdf",dpi=400)