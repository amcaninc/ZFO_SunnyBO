# model_ZFO.jl
# Sunny modelling for ZnFe2O4 system
# Last updated 5/25/2026

function model_ZFO(parameter; SUN = false)
    println("model_ZFO in model_ZFO.jl function called")
    
    ### Manually building crystal
    # latvecs = lattice_vectors(8.43370, 8.43370, 8.43370, 90, 90, 90)
    # positions = [[-0.12408 -0.62408 1.12408]]
    # MyCrystal = Crystal(latvecs, positions, 216)
    # view_crystal(MyCrystal)

    # Now build a system with 1×1×2 chemical cells. Fill magnetic moments according to the MCIF file.
    # sys = System(cryst, dims=(1,1,2), [1=>Moment(s=5/2,g=2)], :dipole, seed=0)
    # sys = System(MyCrystal, dims=(1,1,2), [1=>Moment(s=5/2,g=2)], :dipole)
    # Sunny.set_dipoles_from_mcif!(sys, joinpath(@__DIR__, "..", "Ic42mmW1andmW4.cif"))

    ### Crystal from cif
    cif=joinpath(@__DIR__, "..", "..", "data", "ZnFe2O4_Fd-3m.cif")
    MyCrystal = Crystal(cif; symprec=1e-3)
    MyCrystal = subcrystal(MyCrystal, "Fe")
    # view_crystal(cryst)

    sys = System(MyCrystal, [1 => Moment(; s=5/2, g=2)], :dipole, dims=(1,1,1))
    # plot_spins(sys) 

    J1 = parameter[1]
    J2 = parameter[2] # Primary interaction with allowed DM
    J3 = parameter[3]
    J4 = parameter[4]
    # D = parameter[5] # Uncomment for DM interaction
    ### Exchange interactions for Fd-3m cif
    # commented out DM interactions for J1, J2    
    set_exchange!(sys, J1, Bond(1, 2, [0,0,0]))  # J1
    # set_exchange!(sys, J1*I+dmvec([D,-D,0]), Bond(1, 2, [0,0,0]))  # J1 with DM
    set_exchange!(sys, J2, Bond(1, 7, [0,0,0]))  # J2
    # set_exchange!(sys, J2*I+dmvec([E,F,-E]), Bond(1, 7, [0,0,0]))  # J2 with DM (3 independent DM interactions in the system total)
    set_exchange!(sys, J3, Bond(1, 3, [1,0,0]))  # J3a -- Careful here!
    set_exchange!(sys, J4, Bond(1, 3, [0,0,0])); # J3b -- And here!

    # IMPORTANT you need (1,1,1) supercell for SCGA, do not resize

    sys, MyCrystal
end 
