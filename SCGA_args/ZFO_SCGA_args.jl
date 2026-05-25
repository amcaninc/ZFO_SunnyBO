# ZFO_SCGA_args.jl
# main file for generating ZnFe2O4 static intensity plots by specifying input parameters
# last updated 5/25/2026

using Sunny, CairoMakie
using Revise

includet("src/model_ZFO_args.jl")

input = Float64[]
sysname = "ZnFe2O4"

# manually specify cif file here if running line-by-line, otherwise use ARGS
ciffile = "ZnFe2O4_Fd-3m.cif"

### USAGE
# This script is for generating a ZFO static intensities plot with args input parameters

# Input parameters can be inputted directly from the command line as shown below
# However, this means load_fittingdata is called every time you do this, which takes a while to run

# To run script line-by-line and save time, you can instead specify the parameters manually in this file
# Currently the data is re-loaded every time the whole file is run because of load_fittingdata, but this only needs to be run once
# Uncomment the below section and run load_fittingdata once. You should be able to specify the input again and re-run as needed

### Input
# for Fd-3m lattice, params=[J1,J2,J3a,J3b], adding a fifth parameter adds DM interaction on J1
# for F-43m, params=[J1,J2,J3,J4a,J4b]
input = [
    1,
    0.0328,
    0.0857,
    0.0792,
    # 0.05       
]

### ARGS parsing
if (length(ARGS) == 5 || length(ARGS) == 6) && (ARGS[1] == "F-43m" || ARGS[1] == "Fd-3m")
    input = parse.(Float64, ARGS[2:end])

    # cif file
    ciffile = sysname * "_" * ARGS[1] * ".cif"
else
    println("Usage: julia ZFO_SCGA.jl <spacegroup> <params>")
    exit(1)
end

### Loading and generating SCGA model
# A dataset has to be loaded in with load_fittingdata, but not for fitting in any way
    # To create a 3D HKL grid, the binstart, binend, and numbins are obtained from the scattering data
    # The SCGA model 3D grid is then created with the same binning parameters
data_params = load_fittingdata()

res = model_ZFO(input, data_params, ciffile) # S(Q)


#### draw_fig
# draw an image that displays static intensity H-K sslices at L=0,0.5,1
# attaches string onto image filename and label
label = "label"

# l binwidth adds together intensities around each point
# for example, if you set l_width=0.2, you get plots around L=[-0.1,0.1], [0.4,0.6], and [0.9,1.1]
# will not be exact, depends on if bin centers fall within range
l_width = 0.2

draw_fig(res, data_params, label, input, l_width)
# Image should be saved in a folder called images in this main directory