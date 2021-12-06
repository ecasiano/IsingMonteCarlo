# Classical Monte Carlo Simulation for Ising Model

"--------------------- Function Definitions (start) ---------------------------"

function create_ising_lattice(L)
    "Creates an L x L of Ising spins (all pointing up; +1)"
    ones(Int64,L,L)
end

"------------------------------------------------------------------------------"

function spin_flip!(ising_lattice, L, T)
    "Chooses a spin randomly and proposes to flip it"

    # Sample row,column indices of spin
    x = rand(1:L)
    y = rand(1:L)

    # Compute energy of spin and neighboring bonds before update
    E_old = 0
    E_new = 0
    if x != L
        E_old -= ising_lattice[x,y]*ising_lattice[x+1,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x+1,y]
    else
        E_old -= ising_lattice[x,y]*ising_lattice[1,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[1,y]
    end

    if y != L
        E_old -= ising_lattice[x,y]*ising_lattice[x,y+1]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,y+1]
    else
        E_old -= ising_lattice[x,y]*ising_lattice[x,1]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,1]
    end
    
    if x != 1
        E_old -= ising_lattice[x,y]*ising_lattice[x-1,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x-1,y]
    else
        E_old -= ising_lattice[x,y]*ising_lattice[L,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[L,y]
    end

    if y != 1
        E_old -= ising_lattice[x,y]*ising_lattice[x,y-1]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,y-1]
    else
        E_old -= ising_lattice[x,y]*ising_lattice[x,L]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,L]
    end
    
    # Calculate energy difference between new and old configurations
    E_flip = E_new - E_old

    # Metropolis sampling
    R = exp(-E_flip/T)
    #print(R)
    if rand() < R
        ising_lattice[x,y] *= (-1)
    end

    return nothing

end

"------------------------------------------------------------------------------"

function get_energy(ising_lattice, L)
    "Computes energy of LxL Ising configuration"

    E = 0
    for i in 1:L
        for j in 1:L
            # Accumulate 'horizontal' bond
            if i != L
                E -= ising_lattice[i,j]*ising_lattice[i+1,j] 
            else
                E -= ising_lattice[i,j]*ising_lattice[1,j]
            end
             
            # Accumulate 'vertical' bond
            if j != L
                E -= ising_lattice[i,j]*ising_lattice[i,j+1] 
            else
                E -= ising_lattice[i,j]*ising_lattice[i,1]
            end
        end
    end
    return E
end

"------------------------------------------------------------------------------"

function get_magnetization(ising_lattice, L)
    "Computes magnetization of LxL Ising configuration"

    M = 0
    for i in 1:L
        for j in 1:L
            M += ising_lattice[i,j]
        end
    end
    return M
end

"--------------------- Function Definitions (end) -----------------------------"

"---------------------------- Main (start) ------------------------------------"

# Ising Lattice Parameters (want to make command line parameters later)
L = 8
T = 1.0
K_B = 1

# Simulation parameters    (want to make command line parameters later)
sweep = L*L
bins_wanted = 1024*8
bin_size = 2500

# Initialize Ising lattice
ising_lattice = create_ising_lattice(L)

# Initialize accumulators of quantities to be measured
E = 0
M = 0
E_squared = 0
M_squared = 0 

# Open file to write data to
open("L_$(L)_T_$(T).dat","w") do f
    write(f, "# L=$(L), T=$(T) \n")
    write(f, "# E      M        E^2        M^2\n")

# Monte Carlo loop
let m = 0              
    let bins_written = 0
        let measurement_ctr = 0
            while bins_written < bins_wanted

                # Update iteration counter
                m += 1

                # Propose updates (just a spin flip in this case)
                spin_flip!(ising_lattice,L,T)

                # Measure quantities of interest
                if (m%sweep==0)

                    global E += get_energy(ising_lattice,L)
                    global M += get_magnetization(ising_lattice,L)

                    global E_squared += get_energy(ising_lattice,L)^2
                    global M_squared += get_magnetization(ising_lattice,L)^2

                    measurement_ctr += 1

                    if (measurement_ctr == bin_size)

                        # Write data if accumulator has 'bin_size no. of samples
                        write(f, "$(E/bin_size)    $(M/bin_size)     $(E_squared/bin_size)     $(M_squared/bin_size)\n")

                        bins_written += 1

                        # Reset accumulators and measurement ctr
                        global E = 0
                        global M = 0
                        global E_squared = 0
                        global M_squared = 0
                        measurement_ctr = 0

                        
                    end # end of writing-to-disk if-statement
                end # end of perform-measurement-and-accumulate if-statemenet
            end # end of main Monte Carlo (while) loop
        end # end of let measurement_ctr
    end # end of let bins_written
end # end of let m
end