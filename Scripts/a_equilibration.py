# Calculates running average of Ising energy
import numpy as np
import matplotlib.pyplot as plt

# Load data
# data = np.loadtxt("Data/L_16_T_1.0.dat")           # beta=1
data = np.loadtxt("Data/L_16_T_0.3333333333333333.dat")  # beta=3

# Retrieve energy data
E = data[:,0]

# Calculate the energy per spin for each bin
L = 16
total_sites = L**2
energy_per_spin = E/total_sites

# Compute running average of e (energy per sping)
energy_per_spin_moving_average = np.cumsum(energy_per_spin) 
sample_number = np.cumsum(np.ones_like(energy_per_spin))
energy_per_spin_moving_average=np.divide(energy_per_spin_moving_average,sample_number)

# Plot
plt.plot(sample_number,energy_per_spin_moving_average,label=r'$\beta=%.1f$,L=16'%(3.0))
plt.ylabel(r'$\langle e \rangle$',labelpad=0.0)
plt.xlabel("Sample Number");
plt.legend(frameon=False)
#plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig("Figures/a_equilibration_beta_3.0.pdf",dpi=400)