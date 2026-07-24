# model_ZFO.jl
# Sunny modelling for ZnFe2O4 system
# Last updated 7/23/2026

# Set the four ZFO exchange couplings on `sys` from a parameter vector
# p = [J1, J2, J3a, J3b]. Adding more parameters in BO automatically adds Dzyaloshinskii–Moriya (DM) interactions to exchange bonds

# this is used twice: once for the raw (proposed) couplings and once for the
# Tc-normalized couplings.
function set_ZFO_exchanges!(sys, p)
    if length(p) == 4
        set_exchange!(sys, p[1], Bond(1, 2, [0,0,0]))  # J1
        set_exchange!(sys, p[2], Bond(1, 7, [0,0,0]))  # J2
        set_exchange!(sys, p[3], Bond(1, 3, [1,0,0]))  # J3a
        set_exchange!(sys, p[4], Bond(1, 3, [0,0,0]))  # J3b

        # Currently, assigning more parameters adds DM interactions to the four nearest-neighbor exchange interactions
        # However, it is doubtful that these interactions (at their current [-0.01 - 0.01] range) are impactful on the fit
        # TODO reevaluate if these are necessary to include or if including further interactions improves fit

    elseif length(p) == 5 # DM interaction on J1
        D = p[5]
        set_exchange!(sys, p[1]*I+dmvec([D,-D,0]), Bond(1, 2, [0,0,0]))  # J1 with DM
        set_exchange!(sys, p[2], Bond(1, 7, [0,0,0]))  # J2
        set_exchange!(sys, p[3], Bond(1, 3, [1,0,0]))  # J3a
        set_exchange!(sys, p[4], Bond(1, 3, [0,0,0]))  # J3b
    elseif length(p) == 6 # two independent DM interactions on J2
        E = p[5]
        F = p[6]
        set_exchange!(sys, p[1], Bond(1, 2, [0,0,0]))  # J1
        set_exchange!(sys, p[2]*I+dmvec([E,F,-E]), Bond(1, 7, [0,0,0]))  # J2 with DM (2 independent DM interactions in the system total)
        set_exchange!(sys, p[3], Bond(1, 3, [1,0,0]))  # J3a
        set_exchange!(sys, p[4], Bond(1, 3, [0,0,0]))  # J3b
    elseif length(p) == 7 # three independent DM interactions, one on J1 two on J2
        D = p[5]
        E = p[6]
        F = p[7]
        set_exchange!(sys, p[1]*I+dmvec([D,-D,0]), Bond(1, 2, [0,0,0]))  # J1 with DM
        set_exchange!(sys, p[2]*I+dmvec([E,F,-E]), Bond(1, 7, [0,0,0]))  # J2 with DM (3 independent DM interactions in the system total)
        set_exchange!(sys, p[3], Bond(1, 3, [1,0,0]))  # J3a
        set_exchange!(sys, p[4], Bond(1, 3, [0,0,0]))  # J3b
    end
    return sys
end

# Model is set to a normalized energy scale by first establishing the ordering temperature (10K), determining what the
# proper minimum eigenvalue should be, finding the minimum eigenvalue of the Fourier exchange matrix J(q) 
# given the sampled parameters from Bayesian Optimization, and scaling the exchange matrix
# by the appropriate scaling factor "a" to ensure the model meets this ordering temperature

# epsilon_min = min over q of the smallest eigenvalue of the Fourier exchange matrix,
# This sets the mean-field ordering temperature via kB*Tc = -(S(S+1)/3)*epsilon_min.
# For an ordering (antiferromagnetic) system epsilon_min < 0.

function meanfield_epsilon_min(sys; dq = 0.25)
    qs = Sunny.make_q_grid(sys, dq)
    return minimum(eigmin(Sunny.fourier_exchange_matrix(sys; q)) for q in qs)
end

# `target_Tc` : mean-field ordering temperature (K) that every parameter set is normalized to
function model_ZFO(parameter; SUN = false, target_Tc = 10.0, dq = 0.25)
    println("model_ZFO in model_ZFO.jl function called")

    ### Crystal from cif
    cif=joinpath(@__DIR__, "..", "..", "data", "ZnFe2O4_Fd-3m.cif")
    MyCrystal = Crystal(cif; symprec=1e-3)
    MyCrystal = subcrystal(MyCrystal, "Fe")
    # view_crystal(cryst)

    sys = System(MyCrystal, [1 => Moment(; s=5/2, g=2)], :dipole, dims=(1,1,1))
    # plot_spins(sys) 

    ### Exchange interactions for Fd-3m cif
    set_ZFO_exchanges!(sys, parameter)

    ### Dynamic mean-field Tc normalization
    # finds target minimum eigenvalue given the target temperature Tc
    εmin = meanfield_epsilon_min(sys; dq)
    εmin_target = -3 * (target_Tc * meV_per_K) / ((5/2) * (5/2 + 1))

    εmin_floor = 1e-6  # meV; below this there is no clear ordering instability
    if εmin > -εmin_floor
        @warn "meanfield_epsilon_min = $εmin ≥ 0: no clear ordering mode; " *
              "skipping Tc normalization (a = 1)."
        a = 1.0
    else
        a = εmin_target / εmin
    end

    # Apply the scaled couplings. By linearity this makes the mean-field Tc equal
    # to target_Tc, giving a consistent energy scale for the SCGA model.
    set_ZFO_exchanges!(sys, parameter .* a)

    # IMPORTANT you need (1,1,1) supercell for SCGA, do not resize

    return sys, MyCrystal, a
end
