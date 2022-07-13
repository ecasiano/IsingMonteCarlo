# Classical Monte Carlo Simulation for Ising Model
using Random

"--------------------- Function Definitions (start) ---------------------------"

function create_ising_lattice(L)
    "Creates L x L of randomly orientedIsing spins"
    #ones(Int64,L,L)
    rand([-1, 1], L, L)
end

"------------------------------------------------------------------------------"

function spin_flip!(ising_lattice, L, T, do_nnn=false)
    "Chooses a spin randomly and proposes to flip it"

    # Sample row,column indices of spin
    x = rand(1:L)
    y = rand(1:L)

    # Compute energy between nearest-neighbor spin bonds before & after update
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

    if do_nnn==true
        # Compute energy between next-nearest-neighbor bonds before and after update
        if x == L
            bottom = 1
        else
            bottom = x+1
        end

        if y == L
            right = 1
        else
            right = y+1
        end
        
        if x == 1
            top = L
        else
            top = x-1
        end

        if y == 1
            left = L
        else
            left = y-1
        end

        E_old -= ising_lattice[x,y]*ising_lattice[bottom,right]
        E_old -= ising_lattice[x,y]*ising_lattice[bottom,left]
        E_old -= ising_lattice[x,y]*ising_lattice[top,left]
        E_old -= ising_lattice[x,y]*ising_lattice[top,right]

        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[bottom,right]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[bottom,left]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[top,left]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[top,right]
    end

    # Calculate energy difference between new and old configurations
    E_flip = E_new - E_old

    # Metropolis sampling
    if E_flip < 0
        ising_lattice[x,y] *= (-1)
    elseif rand() < exp(-E_flip/T)
        ising_lattice[x,y] *= (-1)
    end
    
    return nothing

end

"------------------------------------------------------------------------------"

function get_energy(ising_lattice, L, do_nnn=false)
    "Computes energy of LxL Ising configuration"

    E = 0.0
    E_nnn = 0.0
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

            if do_nnn==true

                # Compute energy between next-nearest-neighbor bonds before and after update
                if i == L
                    bottom = 1
                else
                    bottom = i+1
                end

                if j == L
                    right = 1
                else
                    right = j+1
                end

                if i == 1
                    top = L
                else
                    top = i-1
                end

                if j == 1
                    left = L
                else
                    left = j-1
                end

                E_nnn -= ising_lattice[i,j]*ising_lattice[bottom,right]
                E_nnn -= ising_lattice[i,j]*ising_lattice[top,right]

            end

        end
    end

    return E + E_nnn
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
L = 10
T = 2.25
K_B = 1
seed = 0
Random.seed!(seed)

# Simulation parameters    (want to make command line parameters later)
sweep = L*L
bins_wanted = 100000
bin_size = 1

equilibration_steps = 100000

# number of Monte Carlo sweeps to skip before performing a measurement
skip = 1

# Initialize Ising lattice
ising_lattice = create_ising_lattice(L)

# Initialize accumulators of quantities to be measured
E = 0
M = 0
E_squared = 0
M_squared = 0 

# Set type of interaction
# false: nearest-neighbor interaction 
# true:  next-nearest-neighbor interaction
do_nnn = true

# Set interaction label for data file
if do_nnn
    interaction_label = "NNN"
else
    interaction_label = "NN"
end

# Open file to write data to
open("L_$(L)_T_$(T)_seed_$(seed)_$(interaction_label).dat","w") do f
    if (seed==0)
    write(f, "# L=$(L), T=$(T) \n")
    write(f, "# E      M        E^2        M^2\n")
    end

# Open file to write configurations to
open("L_$(L)_T_$(T)_spins_seed_$(seed)_$(interaction_label).dat","w") do g
    if (seed==0)
    write(g, "# L=$(L), T=$(T) \n")
    write(g, "# Ising configs (vectorized) \n")
    end

# Monte Carlo loop
let m = 0              
    let bins_written = 0
        let measurement_ctr = 0
            while bins_written < bins_wanted

                # Update iteration counter
                m += 1

                # Propose updates (just a spin flip in this case)
                spin_flip!(ising_lattice,L,T,do_nnn)

                # Measure quantities of interest
                if (m%(sweep*skip)==0 && m>equilibration_steps)

                    global E += get_energy(ising_lattice,L,do_nnn)
                    global M += get_magnetization(ising_lattice,L)

                    global E_squared += get_energy(ising_lattice,L)^2
                    global M_squared += get_magnetization(ising_lattice,L)^2

                    measurement_ctr += 1

                    if (measurement_ctr == bin_size)

                        # Write data if accumulator has 'bin_size no. of samples
                        write(f, "$(E/bin_size)    $(M/bin_size)     $(E_squared/bin_size)     $(M_squared/bin_size)\n")

                        # Write spin configs if accumulator has 'bin_size no. of samples
                        for i in 1:L
                            for j in 1:L
                                write(g,"$(ising_lattice[i,j]) ")
                            end
                        end
                        write(g,"\n")

                        bins_written += 1
                        if (bins_written%1000==0)
                            println("bins_written: ",bins_written)
                        end

                        # Reset accumulators and measurement ctr
                        global E = 0
                        global M = 0
                        global E_squared = 0
                        global M_squared = 0
                        measurement_ctr = 0

                        
                    end # end of writing-to-disk if-statement
                    end # end of writing-spins-to-disk if-statement 
                end # end of perform-measurement-and-accumulate if-statemenet
            end # end of main Monte Carlo (while) loop
        end # end of let measurement_ctr
    end # end of let bins_written
end # end of let m
end
