# loss_ZFO_HKSlices.jl
# Functions for ZFO_SCGA_bopt.jl
# Last updated 7/23/2026

function load_fittingdata() 
    println("load_fittingdata called")
    params, datas = [], []

    # loading data from nxs file
    # TODO: add way to change temperature in ARGS/main when we start using other datasets
    params_tmp, data_tmp = load_nxs(joinpath(@__DIR__, "..", "..", "data", "ZFO1000_3p32meV_11K_HKLE4D.nxs"))
    println("data start:", params_tmp.binstart, "end: ", params_tmp.binend, "nums: ", params_tmp.numbins)

    # selecting specific energy slice
    # this is meant to integrate for all energy bins between [Elow, Ehigh]
    # and the E_target value
    E_low = params_tmp.binstart[4] # currently set to lowest energy, about -3mev
    # E_low = -1.0 
    E_high = 1.0 # meV

    # integrating all data in energy bin
    params_3D, data_3D, E_lowest, E_highest = integrate_energy(params_tmp, data_tmp, E_low, E_high)
    println("Integrated start:", params_3D.binstart, "end: ", params_3D.binend, "nums: ", params_3D.numbins)

    # mask bragg peaks in data (not computed by SCGA)
    params_mask, data_mask = mask_Bragg(params_3D, data_3D)

    push!(params, params_mask)
    push!(datas, data_mask)

    return params, datas, E_lowest, E_highest
end

function integrate_energy(params::BinningParameters, data::Array{Float64,4}, E_low::Float64, E_high::Float64)
    println("integrate_energy called")

    E_binwidth = params.binwidth[4]
    E_binstarts = [params.binstart[4] + (i-1) * E_binwidth for i in 1:params.numbins[4]]
    E_mask = findall(e -> E_low <= e <= E_high, E_binstarts)
    if isempty(E_mask)
        error("No energy bins found between E_low=$E_low and E_high=$E_high. Check your bin boundaries.")
    end

    # E_binstarts[i] is the LEFT edge of bin i. The integrated window spans the
    # left edge of the lowest selected bin to the right edge of the highest:
    #   left edge of bin i  = binstart + (i-1)*w
    #   right edge of bin i = binstart +  i   *w
    E_highest = params.binstart[4] + maximum(E_mask) * E_binwidth       # right edge of top bin
    E_lowest  = params.binstart[4] + (minimum(E_mask) - 1) * E_binwidth # left edge of bottom bin
    # E_lowest  = params.binstart[4] # use this to get the min E val in the dataset
    println("Integrating energy between E_low=$E_lowest and E_high=$E_highest")

    data_window = data[:, :, :, E_mask]
    data_3d = sum(data_window, dims=4) .* E_binwidth

    binstart_new = copy(params.binstart)
    binend_new   = copy(params.binend)
    binwidth_new = copy(params.binwidth)
    binstart_new[4] = E_lowest
    binend_new[4]   = E_highest
    binwidth_new[4] = E_highest - E_lowest

    params_integrated = BinningParameters(binstart_new, binend_new, binwidth_new, params.covectors)

    return params_integrated, data_3d, E_lowest, E_highest
end

function mask_Bragg(params, data)
    # masks (111) and (220) Bragg peaks from data
    # SCGA does not calculate Bragg peak intensity properly, mask for better fitting

    data_masked = copy(data)
    p = params

    # Symmetrically equivalent points for (111): all sign combinations of (±1, ±1, ±1)
    bragg_111 = [(s1*1, s2*1, s3*1) for s1 in [-1,1], s2 in [-1,1], s3 in [-1,1]][:]

    # Symmetrically equivalent points for (220): all permutations of (±2, ±2, 0)
    # In cubic symmetry: (±2,±2,0), (±2,0,±2), (0,±2,±2)
    bragg_220 = [(s1*2, s2*2, 0) for s1 in [-1,1], s2 in [-1,1]][:]
    append!(bragg_220, [(s1*2, 0, s2*2) for s1 in [-1,1], s2 in [-1,1]][:])
    append!(bragg_220, [(0, s1*2, s2*2) for s1 in [-1,1], s2 in [-1,1]][:])

    all_bragg_peaks = vcat(bragg_111, bragg_220) # combines arrays

    # Bin centers for H, K, L axes
    bins_H = [p.binstart[1] + (i - 0.5) * p.binwidth[1] for i in 1:p.numbins[1]]
    bins_K = [p.binstart[2] + (j - 0.5) * p.binwidth[2] for j in 1:p.numbins[2]]
    bins_L = [p.binstart[3] + (k - 0.5) * p.binwidth[3] for k in 1:p.numbins[3]]

    mask_radius = 0.2

    for (pH, pK, pL) in all_bragg_peaks
        # Find bin indices within ±0.1 of each Bragg peak coordinate
        H_idx = findall(h -> abs(h - pH) <= mask_radius, bins_H)
        K_idx = findall(k -> abs(k - pK) <= mask_radius, bins_K)
        L_idx = findall(l -> abs(l - pL) <= mask_radius, bins_L)

        # Skip if peak is outside the binned volume entirely
        isempty(H_idx) && continue
        isempty(K_idx) && continue
        isempty(L_idx) && continue

        # Set all matching bins to NaN across all energy slices
        data_masked[H_idx, K_idx, L_idx, :] .= NaN
    end

    println("Masked $(length(all_bragg_peaks)) Bragg peaks: (111) family + (220) family")
    return params, data_masked
end

function draw_fig(S1, S2, sim, data, params, ID, loss, parameter, E_low, E_high, a; run_name::String="")
    println("draw_fig in loss_ZFO_QEslices.jl called")

    # Bins from each HKL axis from experimental TOF data
    bins_H = range(params[1].binstart[1], params[1].binend[1], params[1].numbins[1])
    bins_K = range(params[1].binstart[2], params[1].binend[2], params[1].numbins[2])
    bins_L = range(params[1].binstart[3], params[1].binend[3], params[1].numbins[3])

    data3d = data[1][:,:,:,1] # TOF data
    sim3d  = sim[1][:,:,:,1] # SCGA simulated data
    fitted3d  = S1[1] .* sim3d .+ S2[1] # Scaling simulated data
    residual3d = data3d .- fitted3d # Residual of experiment - SCGA
    L_vec  = collect(bins_L)
    L_binwidth = step(bins_L)   # intrinsic bin width of the simulation grid

    # Selecting indices of data around specified L bins
    L_targets = [0.0, 0.5, 1.0] # L indices to plot
    # L_binwidth = params[1].binwidth[3]

    n_rows   = length(L_targets)
    panel_px = 300                       # side length (px) of each square panel
    header_h = 78                        # room for the 3-line parameter header
    fig = Figure(size = (3 * panel_px + 180, panel_px * n_rows + header_h),
                 figure_padding = (8, 8, 8, 8))
    l_width = 0.2

for (row, L_val) in enumerate(L_targets)
        # L bin
        # find all bin indices whose center falls between
        # [L_val - l_width/2, L_val + l_width/2]
        L_indices = findall(abs.(L_vec .- L_val) .<= l_width / 2)

        if isempty(L_indices)
            @warn "No bins found within l_width=$l_width of L=$L_val; using nearest bins"
            L_indices = [argmin(abs.(L_vec .- L_val))]
        end

        # selecting slices to plot, should
        data_slice = dropdims(sum(data3d[:, :, L_indices], dims=3), dims=3)
        fitted_slice = dropdims(sum(fitted3d[:, :, L_indices], dims=3), dims=3)
        resid_slice = dropdims(sum(residual3d[:, :, L_indices], dims=3), dims=3)

        L_lo = round(L_vec[first(L_indices)] - L_binwidth / 2, digits=4)
        L_hi = round(L_vec[last(L_indices)]  + L_binwidth / 2, digits=4)

        # color limits for intensities, scaling
        clim      = (0, max(maximum(filter(!isnan, data_slice)), maximum(filter(!isnan, fitted_slice))))
        resid_lim = maximum(abs.(filter(!isnan, resid_slice)))

        # Experimental
        ax1 = Axis(fig[row, 1], title="Experiment  L=$(L_lo)-$(L_hi), E=$(round(E_low, digits=3))-$(round(E_high, digits=3))meV", xlabel="H", ylabel="K",
                   limits=(-4, 4, -4, 4), aspect=DataAspect())
        hm1 = heatmap!(ax1, bins_H, bins_K, data_slice, colorrange=clim)
        Colorbar(fig[row, 2], hm1, label="Intensity", width=12)

        # Model
        ax2 = Axis(fig[row, 3], title="SCGA Model  L=$(L_lo)-$(L_hi)", xlabel="H", ylabel="K",
                   limits=(-4, 4, -4, 4), aspect=DataAspect())
        hm2 = heatmap!(ax2, bins_H, bins_K, fitted_slice, colorrange=clim)
        Colorbar(fig[row, 4], hm2, label="Intensity", width=12)

        # Residual
        ax3 = Axis(fig[row, 5], title="Residual  L=$(L_lo)-$(L_hi)", xlabel="H", ylabel="K",
                   limits=(-4, 4, -4, 4), aspect=DataAspect())
        hm3 = heatmap!(ax3, bins_H, bins_K, resid_slice, colorrange=(-resid_lim, resid_lim), colormap=:bwr)
        Colorbar(fig[row, 6], hm3, label="Data - Model", width=12)
    end

    colgap!(fig.layout, 6)
    rowgap!(fig.layout, 6)

    # Header
    # Base names for the physical couplings; any extra parameters beyond these get a generic "DM<i>" label by default.
    base_jnames = ["J1", "J2", "J3a", "J3b"]
    raw_vals = round.(vec(parameter), digits=4)
    scl_vals = round.(vec(parameter) .* a, digits=4)
    jnames   = [i <= length(base_jnames) ? base_jnames[i] : "DM$(i-4)" for i in eachindex(raw_vals)]
    raw_str  = join(["$(jnames[i])=$(raw_vals[i])" for i in eachindex(raw_vals)], "    ")
    scl_str  = join(["$(jnames[i])=$(scl_vals[i])" for i in eachindex(scl_vals)], "    ")

    header = "ID = $(ID)        χ² = $(round(loss, digits=2))        scale factor a = $(round(a, digits=5))\n" *
             "Input params:        $(raw_str)\n" *
             "Scaled params (a·J): $(scl_str)"

    Label(fig[0, :], header;
          fontsize = 18, font = :bold, justification = :left, halign = :center,
          padding = (0, 0, 10, 6), tellwidth = false)

    ### Saving figure
    imgdir = isempty(run_name) ? joinpath(@__DIR__, "..", "BOrun") :
                                 joinpath(@__DIR__, "..", "images", run_name)
    mkpath(imgdir)
    save(joinpath(imgdir, "SCGA_$(ID).png"), fig)
    return fig
end

# Objective function
# `draw` controls whether the diagnostic figure is drawn/saved. It must be
# false during threaded (parallel) evaluation, because Makie is not thread-safe.
function SqObjective(exparas, ID, params, data, E_low, E_high; draw::Bool=true, run_name::String="")
    println("SqObjective in loss_ZFO_QEslices.jl called")
    parameter = exparas 
    println("exchange parameters: $parameter")

    sys, MyCrystal, a = model_ZFO(parameter; SUN = false)

    measure = ssf_custom((q, ssf) -> real(sum(ssf)), sys)
    kT = 11*meV_per_K
    scga = Sunny.SCGA(sys; measure, kT=kT, dq=0.25)
    dat_cal = []
    res = []

    # should automatically update based on the binstart, end, and numbins previous specified
    grid = q_space_grid(MyCrystal,
        [1,0,0], range(params[1].binstart[1], params[1].binend[1], params[1].numbins[1]), # H
        [0,1,0], range(params[1].binstart[2], params[1].binend[2], params[1].numbins[2]), # K
        [0,0,1], range(params[1].binstart[3], params[1].binend[3], params[1].numbins[3])) # L
    res = intensities_static(scga, grid)
    push!(dat_cal, res.data)    

    χ² = zeros(1)
    S1 = zeros(1)
    S2 = zeros(1)

    i=1
    data_i = dropdims(data[i], dims=4)
    S1[i], S2[i], χ²[i], diff_map = fitSpectrum(data_i, dat_cal[i], 0) #S2 = 0 to indicate no background

    loss = sum(χ²)
    println("Loss=$loss")
    println("S1=$S1")
    println("S2=$S2")
    println("chi^2=$χ²")
    println("scale factor a=$a")

    # Skip plotting when draw=false (e.g. threaded seed evaluation): Makie is
    # not thread-safe, and concurrent GLMakie/OpenGL rendering can segfault.
    if draw
        draw_fig(S1, S2, dat_cal, data, params, ID, loss, parameter, E_low, E_high, a; run_name=run_name)
    end
    # end
    println("SqObjective done.")
    return loss
end