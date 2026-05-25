# ZFO BO with Sunny

This project has two different scripts
- BO_SCGA models the static intensity of ZnFe2O4 using Sunny's SCGA function and runs a Bayesian Optimization loop that compares the fit with TOF neutron data
- SCGA_args allows you to input exchange interaction parameters in-file or on command line ARGS. The script plots static intensity from the calculated SCGA model

## BO_SCGA

### Description
Bayesian Optimization loop that uses Sunny to calculate the static intensity S(Q) from a model ZnFe2O4 system, compares 
the intensity with experimental data, and computes a fit.

### Setup
This relies on a Python wrapper for Julia for Bayesian Optimization located at https://github.com/sakibmatin/BayesOptim.jl.
You will need to add this to Julia with `add https://github.com/sakibmatin/BayesOptim.jl` using the packages manager in the Julia REPL, this package cannot be added normally by name.

Once this package is added, you still need to change some things. 
To use a manually-picked starting point, find your .julia folder (on Windows should be C:/Users/user/.julia by default)
Replace the Fit.jl file in the BayesOptim package with the file in the BO_SCGA directory


## SCGA_args

### Description
Creates an image of the static intensity for manually-entered Heisenberg exchange interactions

### Usage
`julia ZFO_SCGA_args.jl <spacegroup> <params>`