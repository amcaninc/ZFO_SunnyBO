module Bopt
using Pkg
# Pkg.build("PyCall")
using PyCall
using Random
using OrderedCollections

include("SafetyChecks.jl")
include("Sampling.jl")
include("Fit.jl")

export Fit, latin_hypercube

end
