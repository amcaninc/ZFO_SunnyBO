# ZnFe2O4 Bayesian Optimization with Sunny

This project has three different subprojects:
- BO_SCGA, the main project that uses Bayesian Optimization (Bopt) to model the static intensity of ZnFe2O4 using Sunny's SCGA (self-consistent Gaussian approximation) function. The main file runs a Bayesian Optimization loop that computes a fit between the SCGA model and TOF neutron data
- Bopt, the Bayesian Optimization model used by the main project BO_SCGA. 
- SCGA_args allows you to input exchange interaction parameters in-file or on command line ARGS. The script plots static intensity from the calculated SCGA model

## BO_SCGA

### Description
Bayesian Optimization loop that uses Sunny to calculate the static intensity S(Q) from a ZnFe2O4 system modelled with SCGA, compares the intensity with experimental data, and computes a fit.

#### ZFO_SCGA_bopt.jl (main file)
usage: `julia ZFO_SCGA_bopt.jl -t (num threads on your CPU you want to use)`
for example, if you wanted to use 8 CPU threads, `julia ZFO_SCGA_bopt.jl -t 8`
Threaded evaluations only come from evaluating multiple starting points, NOT the actual BO process. Slightly speeds up Latin hypercube sampling or other multi-point initial seeding to explore parameter region before BO
To change the parameters being evaluated, parameter ranges, BO filenames, and initial point seeding, modify this file
Ensure that when adding/removing parameters, the number of parameters used and their names are consistent across the file and accounted for in all functions

#### src/loss_ZFO_HKSlices.jl
Loads in experimental data from .nxs file, integrates experimental data across a defined energy range (4D HKLE -> 3D HKL), masks atomic Bragg points from data/model, draws figures comparing experiment with model, objective function with SCGA modelling, calculates static intensity from SCGA model
To change experimental data NeXuS file read, energy integration range, masking radius for atomic Bragg points, or SCGA temperature (K -> meV), modify this file

#### src/model_ZFO.jl
sets exchange parameters from BO to a Sunny crystal system, normalizes energy scale by determining calculated minimum eigenvalue based on ordering temperature (10K)
To change model's .cif file or to control individual exchange interactions, check this file
NOTE: You may want to verify that the exchange interactions are being set to the nearest neighbor bonds of your model. To check, load your cif file in a separate Julia file, set up a Sunny system, and run view_cryst(::Crystal)

#### src/funcFitSpectrum.jl
Calculates scale factor and fit value between experiment and model

## Bopt

### Desciption
Bayesian Optimization project that defines the process used in `BO_SCGA/ZFO_SCGA_bopt.jl`

#### Bopt.jl
main file

#### Fit.jl
Main driver and function called for performing Bayesian Optimization in `BO_SCGA/ZFO_SCGA_bopt.jl`. Py-call front-end to `Bopt.py`. Includes objective function, defines GP surrogate, ARD length scales, parallel pre-seeded point evaluation, and exploration rate.

#### Bopt.py
Python back-end, defines surrogate GP, expected improvement acquisition function, and writes csv to track each point evaluated by BO and its fit score.

#### Sampling.jl
Latin-hypercube sampling to find equidistant points for evaluation within the BO parameter bounds. Creates points for initial exploration of the parameter space.

#### SafetyChecks.jl
Sets safety checks and warnings for BO, currently not used for much

## SCGA_args

### Description
Separate project to create an image of the static intensity for manually-entered Heisenberg exchange interactions

usage: `julia ZFO_SCGA_args.jl <spacegroup> <params>`
