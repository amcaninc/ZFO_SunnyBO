# ZFO_SCGA_bopt.jl
# main file for BO loop
# last updated: 7/23/2026

using DrWatson
using Sunny, CairoMakie, LinearAlgebra, DataFrames, CSV, Suppressor, Statistics
using OrderedCollections, StaticArrays, Random 
include(joinpath(@__DIR__, "..", "Bopt", "src", "Bopt.jl")) # Current BO package modified
using .Bopt   # uses the module just defined above
using GLMakie
using PyCall, Conda
using Revise

### Only need to run these the first time you run this script to add packages to Conda
# Conda.add("scipy")
# Conda.add("scikit-learn")
# Conda.add("pandas")

includet("src/model_ZFO.jl")
includet("src/loss_ZFO_HKSlices.jl")
includet("src/funcFitSpectrum.jl")

### Parameter bounds
bounds = OrderedDict(
    # "J1" => 1.0 # FIXED
    "J2" => [-1, 1], # J2
    "J3a" => [-1, 1], # J3
    "J3b" => [-1, 1], # J4

    # DM Interactions - may not be useful for fits? check length scales, maybe just start with Heisenberg
    # "DM1" => [-0.01, 0.01], # DM1
    # "DM2" => [-0.01, 0.01], # DM2
    # "DM3" => [-0.01, 0.01], # DM3
    
    # DM range : -0.01 to 0.01
)

max_iter = 50

### Run Name
# Change this string to change the name of the run
#   writes the log as "<run_name>_Boptlog.csv"
#   saves figures to BO_SCGA/images/<run_name>/ (created automatically)
# Leave "" to get default names.
# Reusing an existing run_name resumes that BO session where it left off.
run_name = "07232026_3DM"

# loading in parameters, energy range
fit_params, datas, E_low, E_high = load_fittingdata()


function Objective(params)
    # Convert Bopt parameter dictionary to Sunny model parameters. 
    println("Objective in main function called")

    R1  = 1
    R2  = get(params, "J2", 0.0)
    R3  = get(params, "J3a", 0.0)
    R4  = get(params, "J3b", 0.0)

    # Adding Dzyaloshinskii–Moriya (DM) interactions to exchange bonds
    # May or may not actually improve fits, for simplicity start with just exchange interactions
    # R5  = get(params, "DM1", 0.0)
    # R6  = get(params, "DM2", 0.0)
    # R7  = get(params, "DM3", 0.0)

    ID  = get(params, "ID", 0)
    # During threaded seed evaluation, Fit sets "draw" => false because Makie
    # is not thread-safe. Defaults to true for the (sequential) BO loop.
    draw = get(params, "draw", true)

    # set DM to 0, allow other 5 parameters to vary

    # Change experas to reflect how many parameters you are using
    exparas = [R1 R2 R3 R4] # J1, J2, J3, J4
    # exparas = [R1 R2 R3 R4 R5 R6 R7] # J1, J2, J3, J4, DM1, DM2, DM3

    # Compute the Loss function.
    Loss = SqObjective(exparas, ID, fit_params, datas, E_low, E_high; draw=draw, run_name=run_name)
    Loss
    return -Loss
end

# There are multiple ways to start/resume a BO run

### Seed points
# Generate a space-filling set of starting points over `bounds` with Latin Hypercube Sampling. 
# `n_seed` points, reproducible via the seeded rng in MersenneTwister.
# if you want to generate 10 initial points across the bounds for BO to sample the space, use this method
n_seed = 10
initial_points = latin_hypercube(bounds, n_seed; rng = MersenneTwister(1))

### Start with pre-defined seeded points to skip calculations
# manually enter each parameter, the BO will run these points first before exploring parameter space
# (each Dict must contain every key in `bounds`; a single-element vector is one starting point):
# initial_points = [
#     OrderedDict("J2" => 0.0328, "J3a" => 0.0857, "J3b" => 0.0792),
#     OrderedDict("J2" => -0.5,   "J3a" => 0.2,    "J3b" => 0.0),
# ]


### Optional: pre-evaluated objective values
# Provide these so BO injects the points WITHOUT re-running Objective, saves time if you have pre-evaluated these points
# in a separate run and already know their fit values
# IMPORTANT: same sign convention Fit uses internally, i.e. the value returned
# by `Objective` (which is -Loss). One entry per point.
# initial_obj = [ -Loss1, -Loss2, -Loss3 ]

# For faster evaluation of multiple starting points, use threads
# NOTE: launch Julia with multiple threads for this to actually parallelize,
# e.g.  julia -t 8   (or set the env var JULIA_NUM_THREADS=8).
# Seed evaluations run with "draw" => false (Makie is not thread-safe); the
# BO loop still draws figures as before.
# Does not speed up main BO loop, only works on initial guesses evaluated simultaneously
#
# Fit returns the fitted ARD length scales (one per parameter). A SMALL length
# scale means the fit is sensitive to that parameter; a LARGE one means it is
# comparatively unimportant. A value pinned at the upper bound (1e3) means the
# parameter looks irrelevant given the data (raise length_scale_bounds in
# Bopt/src/Fit.jl to resolve it further).
length_scales = @time Fit(Objective, bounds, max_iter; initial_points=initial_points, parallel_seeds=true, log_name=run_name)

# ...to inject already-evaluated points (no re-evaluation, no threading needed):
# length_scales = @time Fit(Objective, bounds, max_iter; initial_points=initial_points, initial_obj=initial_obj)

# ...to evaluate seeds one at a time (keeps seed figures; needed if Objective
#    is not thread-safe):
# length_Scales = @time Fit(Objective, bounds, max_iter; initial_points=initial_points, parallel_seeds=false)

println("Fitted length scales: ", length_scales)