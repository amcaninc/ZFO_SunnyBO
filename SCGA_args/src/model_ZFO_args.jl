# functions for ZFO_SCGA_args.jl
# last updated 5/28/2026

function load_fittingdata(E_low, E_high)
    println("load_fittingdata in model_ZFO_args.jl called")
    
    # loading data from nxs file
    # TODO: add way to toggle between 11K and 15K in ARGS
    file_path = joinpath(@__DIR__, "..", "..", "data", "ZFO1000_3p32meV_11K_HKLE4D.nxs")
    params, data = load_nxs(file_path)
    println("data start:", params.binstart, "end: ", params.binend, "numbins: ", params.numbins)

    # Integrate data within specified energy bounds
    params_3D, data_3D, E_lowest, E_highest = integrate_energy(params, data, E_low, E_high)
    println("Integrated start:", params_3D.binstart, "end: ", params_3D.binend, "nums: ", params_3D.numbins)
    return params_3D, data_3D, E_lowest, E_highest
end

function integrate_energy(params::BinningParameters, data::Array{Float64,4}, E_low::Float64, E_high::Float64)
    println("integrate_energy called")

    # Find which bin indices fall within [E_low, E_high]
    # look at start of each bin, exclude all bins where binstart > target energy
    E_binwidth = params.binwidth[4]
    E_binstarts = [params.binstart[4] + (i-1) * E_binwidth for i in 1:params.numbins[4]]
    # filtering based on E_high
    E_mask = findall(e -> E_low <= e <= E_high, E_binstarts)
    if isempty(E_mask)
        error("No energy bins found between E_low=$E_low and E_high=$E_high. Check your bin boundaries.")
    end

    # calculate highest energy value in the highest included bin
    E_highest = params.binstart[4] + maximum(E_mask) * E_binwidth
    E_lowest = params.binstart[4] + minimum(E_mask) * E_binwidth

    println("Integrating energy between E_low=$E_lowest and E_high=$E_highest")

    data_window = data[:, :, :, E_mask]
    # selects all H, K, L indices but only the energy indices in E_mask.

    data_3d = sum(data_window, dims=4) .* E_binwidth
    # sum(..., dims=4) collapses the energy axis by adding across it.
    # this gives an integral between [Elow, Ehigh] multiplying by dE

    binstart_new  = copy(params.binstart)
    binend_new    = copy(params.binend)
    binwidth_new  = copy(params.binwidth)
 
    binstart_new[4] = E_lowest
    binend_new[4]    = E_highest
    binwidth_new[4] = E_highest - E_lowest

    params_integrated = BinningParameters(binstart_new, binend_new, binwidth_new, params.covectors)

    return params_integrated, data_3d, E_lowest, E_highest
end

function model_ZFO(input, data_params, ciffile; SUN = false)
    println("model_ZFO in model_ZFO_args.jl function called")

    cif = joinpath(@__DIR__, "..", "..", "data", ciffile)
    cryst = Crystal(cif; symprec=1e-3)
    cryst = subcrystal(cryst, "Fe")
    # view_crystal(cryst)

    sys = System(cryst, [1 => Moment(; s=5/2, g=2)], :dipole, dims=(1,1,1))

    if ciffile == "ZnFe2O4_Fd-3m.cif"
        if length(input) == 4 
            set_exchange!(sys, input[1], Bond(1, 2, [0,0,0]))  # J1
            set_exchange!(sys, input[2], Bond(1, 7, [0,0,0]))  # J2
            set_exchange!(sys, input[3], Bond(1, 3, [1,0,0]))  # J3a
            set_exchange!(sys, input[4], Bond(1, 3, [0,0,0])); # J3b
        elseif length(input) == 5 # DM interaction on J1
            set_exchange!(sys, J1*I+dmvec([input[5],-input[5],0]), Bond(1, 2, [0,0,0])) # J1 with DM
            set_exchange!(sys, input[2], Bond(1, 7, [0,0,0]))  # J2
            set_exchange!(sys, input[3], Bond(1, 3, [1,0,0]))  # J3a
            set_exchange!(sys, input[4], Bond(1, 3, [0,0,0])); # J3b
        end
    elseif ciffile == "ZnFe2O4_F-43m.cif"
        # TODO add DM interaction compatability
        set_exchange!(sys, input[1], Bond(2, 3, [1,0,0]))  # J1
        set_exchange!(sys, input[2], Bond(1, 2, [0,0,0]))  # J2
        set_exchange!(sys, input[3], Bond(1, 7, [0,0,0]))  # J3
        set_exchange!(sys, input[4], Bond(1, 3, [1,0,0])); # J4
        set_exchange!(sys, input[4], Bond(1, 3, [0,0,0])); # J5
    else
        error("Cif file not found, ensure input is correct and file is in SCGA_args/data")
    end

    randomize_spins!(sys)
    m = minimize_energy!(sys; maxiters=10000);
    if m == -1
        println("Energy minimiztion failed for parameters: $params")
        return 1e6
    end
    
    measure = ssf_custom((q, ssf) -> real(sum(ssf)), sys)
    kT = 11*meV_per_K
    scga = Sunny.SCGA(sys; measure, kT=kT, dq=0.25)

    grid = q_space_grid(cryst,
        [1,0,0], range(data_params.binstart[1], data_params.binend[1], data_params.numbins[1]), # H
        [0,1,0], range(data_params.binstart[2], data_params.binend[2], data_params.numbins[2]), # K
        [0,0,1], range(data_params.binstart[3], data_params.binend[3], data_params.numbins[3]) # L
    )

    res = intensities_static(scga, grid) # S(Q)
    return res
end

function draw_fig(sim, data, params, ID, input, l_width, E_low, E_high)
    println("draw_fig in model_ZFO_args.jl called")

    # Derive bin centers from params
    bins_H = range(params.binstart[1], params.binend[1], params.numbins[1])
    bins_K = range(params.binstart[2], params.binend[2], params.numbins[2])
    bins_L = range(params.binstart[3], params.binend[3], params.numbins[3])
        
    data3d = data[:,:,:,1] # TOF data
    sim3d  = sim.data[:,:,:,1]
    L_vec  = collect(bins_L)
    L_binwidth = step(bins_L)   # intrinsic bin width of the simulation grid

    # L slices to show
    L_targets = [0.0, 0.5, 1.0]

    fig = Figure(size = (700, 300 * length(L_targets)), figure_padding = 10)
    H = collect(bins_H)
    K = collect(bins_K)

    for (row, L_val) in enumerate(L_targets)
        # Find all bin indices whose centers fall within
        # [L_val - l_width/2, L_val + l_width/2]
        L_indices = findall(abs.(L_vec .- L_val) .<= l_width / 2)

        if isempty(L_indices)
            @warn "No bins found within l_width=$l_width of L=$L_val; using nearest bins"
            L_indices = [argmin(abs.(L_vec .- L_val))]
        end

        # Integrate intensity over the selected L bins
        data_slice = dropdims(sum(data3d[:, :, L_indices], dims=3), dims=3)
        fitted_slice = dropdims(sum(sim3d[:, :, L_indices], dims=3), dims=3)

        # Compute the actual L range that was integrated
        L_lo = round(L_vec[first(L_indices)] - L_binwidth / 2, digits=4)
        L_hi = round(L_vec[last(L_indices)]  + L_binwidth / 2, digits=4)

        clim_data = (0, maximum(filter(!isnan, data_slice)))
        clim_model = (0, maximum(filter(!isnan, fitted_slice)))

        # Experimental
        ax1 = Axis(fig[row, 1], title="Experiment  L=$(L_lo)-$(L_hi), E=$(round(E_low, digits=3))-$(round(E_high, digits=3))meV", xlabel="H", ylabel="K",
                   limits=(-4, 4, -4, 4))
        hm1 = heatmap!(ax1, H, K, data_slice, colorrange=clim_data)
        Colorbar(fig[row, 2], hm1, label="Intensity") # FIXME

        # Model
        ax2 = Axis(fig[row, 3],
                   title  = "SCGA Model  L=$(L_lo)-$(L_hi)",
                   xlabel = "H", ylabel = "K",
                   limits = (-4, 4, -4, 4))
        hm2 = heatmap!(ax2, H, K, fitted_slice, colorrange=clim_model)
        Colorbar(fig[row, 4], hm1, label="Intensity")
    end

    Label(fig[0, :],
          "$(ID) | params=$(round.(input, digits=4))",
          fontsize=14)

    mkpath(joinpath(@__DIR__, "..", "images"))
    save(joinpath(@__DIR__, "..", "images", "SCGA_$(ID).png"), fig)
    return fig
end