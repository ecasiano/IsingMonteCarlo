# Evaluate average magnetization per spin
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import ellipk
    
# --- Main --- #
alpha=1.0
z_order=0
for L in [12,16,20,24][::-1]:
    alpha -= 0.2
    z_order-=1
    # Set system size
    #L = 16
    total_sites = L**2

    # Load desired L files
    L_filenames = []
    for subdir, dirs, files in os.walk('../../Data/'):
        for filename in files:
            if filename.split('_')[1] == str(L): L_filenames.append(filename)
    L_filenames = sorted(L_filenames)

    # Compute magnetization for all temperatures in Data directory
    M = []
    throw_bins=100
    for file in L_filenames:
        data = np.loadtxt('../../Data/'+file)

        M_mean = np.mean(data[:,1][throw_bins:])
        M.append(M_mean)
    M = np.array(M)

    # Same as above but for e squared
    M2 = []
    throw_bins=100
    for file in L_filenames:

        data = np.loadtxt('../../Data/'+file)

        M2_mean = np.mean(data[:,3][throw_bins:])
        M2.append(M2_mean)
    M2 = np.array(M2)

    # Retrieve temperatures for which we have data and store to array
    temperatures = []
    for filename in L_filenames:
        temperatures.append(float(filename.split('_')[3].replace('.dat','')))
    T = np.array(temperatures)

    # Compute specific heat for all temperatures
    chi = np.divide((M2-np.square(M)),T)/total_sites
    
# COMPUTE ERROR BARS #

    # Exact critical temperature
    Tc_exact = 2/(np.log(1+np.sqrt(2)))
    
    t = (T-Tc_exact)/Tc_exact
    nu = 1.0
    gamma = 1.7

    # Plot magnetic susceptibility vs temperature
    plt.plot(t*L**(1/nu),chi*L**(-gamma/nu),'-o',label=r'$L=%d,T_{\rm{max}}=%.2f$'%(L,T[np.argmax(chi)]),alpha=alpha,zorder=z_order)
#     plt.plot(T,chi,'-',alpha=1/(0.1*L))
    plt.ylabel(r'$\langle \chi / N \rangle L^{-\gamma/\nu}$')
    plt.xlabel(r"$t L^{1/\nu}$");
    plt.axvline(Tc_exact,zorder=-10,color='gray')
    plt.legend(frameon=False)
    
plt.savefig("../Figures/d_collapse.svg",dpi=400)