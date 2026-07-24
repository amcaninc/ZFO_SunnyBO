

"""
    latin_hypercube(interval, n; rng=Random.default_rng())

Generate `n` starting points by Latin Hypercube Sampling (LHS) over the box
defined by `interval` (an `OrderedDict` of `param => [low, high]`). Returns a
`Vector{OrderedDict{String,Float64}}`, one entry per point, in the same key
order as `interval` — ready to pass straight to `Fit` as `initial_points`.

Latin Hypercube Sampling spreads the points so that, for every parameter, each
of `n` equal-width slices of its range contains exactly one sample. That gives
much more even coverage of the search space than independent uniform random
points (which tend to clump and leave gaps)

Pass a seeded generator for a reproducible design, e.g.
`rng = MersenneTwister(1)`.

# Example
```julia
bounds = OrderedDict("J2" => [-1, 1], "J3a" => [-1, 1], "J3b" => [-1, 1])
initial_points = latin_hypercube(bounds, 10; rng = MersenneTwister(1))
Fit(Objective, bounds, max_iter; initial_points=initial_points)
```
"""
function latin_hypercube(interval, n::Integer; rng=Random.default_rng())
    n >= 1 || throw(ArgumentError("n must be >= 1 (got $n)"))
    ks = collect(keys(interval))
    d = length(ks)

    # Stratified samples in [0,1): split each dimension into n equal slices and
    # place one sample in each, then permute the slice order independently per
    # dimension so the columns are decorrelated.
    U = Matrix{Float64}(undef, n, d)
    for j in 1:d
        perm = randperm(rng, n)
        for i in 1:n
            U[i, j] = (perm[i] - 1 + rand(rng)) / n
        end
    end

    # Scale each unit-cube sample to the parameter bounds and package it.
    points = Vector{OrderedDict{String,Float64}}(undef, n)
    for i in 1:n
        pt = OrderedDict{String,Float64}()
        for (j, k) in enumerate(ks)
            lo = interval[k][1]
            hi = interval[k][2]
            pt[String(k)] = lo + U[i, j] * (hi - lo)
        end
        points[i] = pt
    end
    return points
end
