# ZFO_SCGA_bopt.jl
# main file for BO loop
# last updated: 5/24/2026

using DrWatson
using Sunny, CairoMakie, LinearAlgebra, DataFrames, CSV, Suppressor, Statistics
using OrderedCollections, StaticArrays, Random 
using BayesOptim # important - not present in typical Julia add funcionality, get from github, see README
using GLMakie
using PyCall, Conda
using Revise

### IMPORTANT: make sure to update BayesOptim's fit.jl with the version
# provided in the Google drive to properly set the starting point
# for the BO process.

### Only need to run these the first time you run this script to add packages to Conda
# Conda.add("scipy")
# Conda.add("scikit-learn")
# Conda.add("pandas")

includet("src/model_ZFO.jl")
includet("src/loss_ZFO_HKSlices.jl")
includet("src/funcFitSpectrum.jl")

### Parameter bounds
bounds = OrderedDict(
    # "R1" => [1.0, 1.0], # primary exchange interaction J1 fixed to 1
    # "J2" => [-3, 3], # J2/J1
    "J3" => [-3, 3], # J3/J1
    # "J4" => [-2, 2], # J4/J1
    # "DM1" => [-0.01, 0.01], # DM1/J1
    
    # DM range : -0.01 to 0.01
)

max_iter = 1

# loading in parameters, energy range
fit_params, datas, E_low, E_high = load_fittingdata()


function Objective(params)
    # Convert Bopt parameter dictionary to Sunny model parameters. 
    println("Objective in main function called")

    R1  = 1
    R2  = 0.0328
    R3  = get(params, "J3", 0.0)
    R4  = 0.0792
    # R5  = get(params, "DM1", 0.0)

    ID  = get(params, "ID", 0)
    
    # set DM to 0, allow other 5 parameters to vary

    exparas = [R1 R2 R3 R4] # J1, J2, J3, J4, DM

    # Compute the Loss function. 
    Loss = SqObjective(exparas, ID, fit_params, datas, E_low, E_high)
    Loss
    return -Loss
end

start_point = OrderedDict(
    # "J2" => 0.0328,
    "J3" => 0.0857,
    # "J4" => 0.0792,
    # "DM1" => 0.00640,
)
 
@time Fit(Objective, bounds, max_iter; start_point=start_point)