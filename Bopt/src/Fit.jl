

# Fit.jl — Bayesian optimization driver (PyCall front-end to Bopt.py).
#
# `Objective(params)` is called with params::Dict{String,Any} of the form
#   {"ID" => Int, param_name => value, ...} and MUST return the NEGATIVE of the
#   loss (this routine MAXIMIZES, so larger == better fit). During threaded seed
#   evaluation the params Dict also carries "draw" => false (see notes below).
#
# `interval`  : OrderedDict {param_name => (low, high)}. Use an OrderedDict so the
#               column order is stable and matches the initial-point Dicts.
# `max_iter`  : TOTAL number of objective evaluations (initial points included).
#
# Seeding (all optional):
#   initial_points : Vector of Dict(param => value) — one Dict per starting point.
#                    Each Dict must contain every key in `interval`. Omit to start
#                    from the midpoint of `interval`.
#   initial_obj    : Vector of already-computed Objective values, one per
#                    initial point. If given, those points are injected into the
#                    GP directly WITHOUT calling Objective again (values must be
#                    in objective units, i.e. -loss).
#
# Parallelism:
#   parallel_seeds : if true (default) the initial points are evaluated across
#                    Julia threads. Start Julia with `julia -t N` (or set
#                    JULIA_NUM_THREADS=N). The main BO loop is inherently
#                    sequential and is NOT parallelized.
#                    REQUIREMENT: `Objective` must be thread-safe. In particular
#                    Makie plotting is NOT thread-safe, so seed evaluations pass
#                    "draw" => false and the objective must honor it (skip any
#                    figure drawing/saving when params["draw"] == false).
#   rng_seed       : optional Int for reproducible candidate sampling.

function Fit(Objective, interval, max_iter;
             initial_points=nothing, initial_obj=nothing,
             parallel_seeds=true, rng_seed=nothing, log_name=nothing)

    DIR = @__DIR__
    @pyinclude(DIR*"/Bopt.py")

    # Validate bounds on the Julia side (cleaner than calling across PyCall).
    SafetyChecks(interval)

    ks = collect(keys(interval))   # canonical parameter/column order

    # ---- Log filename -------------------------------------------------------
    # `log_name` lets the caller tag this run. When given, the log is written as
    # "<log_name>_Boptlog.csv"; when omitted (nothing or empty) we fall back to
    # the historical default "Bopt_Log.csv" so existing behavior is unchanged.
    # Reusing a name that already has a log resumes that run (see Restart).
    log_file = (isnothing(log_name) || isempty(log_name)) ? "Bopt_Log.csv" :
               "$(log_name)_Boptlog.csv"

    # ---- Detect an existing, resumable log BEFORE doing any expensive work ---
    # `Restart` reads Bopt_Log.csv (if present) and validates that its parameter
    # columns match the current bounds. We compute `is_fresh` HERE, up front, so
    # that when a valid log already exists we can SKIP re-evaluating the seed
    # points entirely (previously the seeds were always re-run, then discarded).
    py"""
    X_restart, y_restart, idx_restart = Restart($interval, $log_file)
    is_fresh = (int(idx_restart[-1][0]) + 1) == 1
    """
    is_fresh = py"is_fresh"::Bool

    # ---- Build the list of seed parameter vectors (ordered by interval keys) ----
    if isnothing(initial_points)
        midpoint = [(interval[k][1] + interval[k][2]) / 2 for k in ks]
        seed_vecs = [midpoint]
    else
        seed_vecs = [[pt[k] for k in ks] for pt in initial_points]
    end
    nseed = length(seed_vecs)

    if !is_fresh
        # Resuming from an existing Bopt_Log.csv: the seed design is already in
        # the log, so do NOT evaluate the initial points again. The Python block
        # below picks up X/y from the restart data.
        println("Resuming from existing Bopt_Log.csv; skipping initial seed evaluation.")
        seed_X = Matrix{Float64}(undef, 0, length(ks))
        seed_Y = Float64[]
    elseif !isnothing(initial_obj)
        # Pre-evaluated: inject directly, no Objective calls, no threading needed.
        length(initial_obj) == nseed ||
            error("initial_obj has $(length(initial_obj)) values but there are " *
                  "$nseed initial point(s).")
        seed_y = collect(float.(initial_obj))
        keep = findall(isfinite, seed_y)
        isempty(keep) &&
            error("All $nseed initial point(s) failed to evaluate; cannot start BO.")
        seed_X = reduce(vcat, (reshape(seed_vecs[j], 1, :) for j in keep))
        seed_Y = collect(seed_y[keep])
    else
        if parallel_seeds && Threads.nthreads() == 1
            @warn "parallel_seeds=true but Julia is running with a single thread. " *
                  "Start Julia with `-t N` (or JULIA_NUM_THREADS=N) to parallelize."
        end
        seed_y = Vector{Float64}(undef, nseed)

        # Evaluate one seed; return NaN on any failure so a single bad point
        # cannot abort the whole batch. Seeds skip plotting (thread-safety).
        eval_seed = function (j)
            params = Dict{String,Any}("ID" => j, "draw" => !parallel_seeds)
            for (i, k) in enumerate(ks)
                params[k] = seed_vecs[j][i]
            end
            try
                v = float(Objective(params))
                isfinite(v) ? v : NaN
            catch e
                @warn "Initial point $j failed to evaluate" exception = (e, catch_backtrace())
                NaN
            end
        end

        if parallel_seeds && nseed > 1
            Threads.@threads for j in 1:nseed
                seed_y[j] = eval_seed(j)
            end
        else
            for j in 1:nseed
                seed_y[j] = eval_seed(j)
            end
        end

        # ---- Drop any failed (non-finite) seeds ----
        keep = findall(isfinite, seed_y)
        isempty(keep) &&
            error("All $nseed initial point(s) failed to evaluate; cannot start BO.")
        seed_X = reduce(vcat, (reshape(seed_vecs[j], 1, :) for j in keep))
        seed_Y = collect(seed_y[keep])
        if length(keep) < 2
            @warn "Only $(length(keep)) usable initial point(s). GP length scales are " *
                  "poorly identified from a single sample; pass more initial_points " *
                  "for better early performance."
        end
    end

    py"""
    bounds      = $interval
    max_iter    = $max_iter
    initial_X   = $seed_X       # 2-D array of seed params (rows = points)
    initial_Y   = $seed_Y       # 1-D array of seed objective values
    rng_seed    = $rng_seed     # int or None
    obj_fn      = $Objective    # Julia objective as a Python callable
    log_file    = $log_file     # CSV filename for this run's log

    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    # --- Gaussian process surrogate -----------------------------------------
    n_dims = len(bounds)
    kernel = Matern(
        length_scale=[1.0] * n_dims,        # ARD: one length scale per parameter
        length_scale_bounds=(1e-03, 1e3),
        nu=2.5,
    )
    GP_model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,                   # center/scale targets so the GP prior behaves
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=30,
    )

    # --- Restart, or load the initial design evaluated on the Julia side -----
    # Reuse the restart data already read above (do not re-read the CSV).
    X, y, idx_list = X_restart, y_restart, idx_restart

    if is_fresh:
        X = np.atleast_2d(np.array(initial_X, dtype=float))
        y = np.array(initial_Y, dtype=float).reshape(-1, 1)
        idx_list = np.arange(1, X.shape[0] + 1).reshape(-1, 1)
        print("Initial design: %d point(s)" % X.shape[0])

        # Log the initial design immediately (survives a crash mid-loop). The GP
        # is not fit yet, so length scales are not available for this first write.
        write_log(log_file, idx_list, X, y, bounds, length_scales=None)
    else:
        print("Resuming from %s: %d existing evaluation(s)." % (log_file, len(y)))
        if len(y) >= max_iter:
            print("Log already has %d >= max_iter=%d evaluation(s); "
                  "nothing to do. Increase max_iter or start a fresh log."
                  % (len(y), max_iter))

    # --- Bayesian optimization loop (sequential) ----------------------------
    # `max_iter` means exactly "total evaluations" (seeds included).
    idx = int(idx_list[-1][0]) + 1
    while len(y) < max_iter:
        print("Bayesian Opt Step ::", idx)
        GP_model.fit(X, y.ravel())

        # Current ARD length scales from this step's GP fit (kernel_ = optimized
        # kernel). Logged at the top of the file below. NOTE: these reflect the
        # GP that is about to PROPOSE this step's point, i.e. fit on all
        # evaluations BEFORE this step; the table written below also includes the
        # new point. (Read straight from kernel_ so no extra fit is done and the
        # BO/RNG stream is unchanged.)
        _ls_step = np.atleast_1d(GP_model.kernel_.length_scale)
        current_ls = [float(v) for v in _ls_step]

        # Exploration parameter: scaled to the objective spread (std of y) and
        # decayed over the budget from explorative -> exploitative.
        sigma = np.std(y)
        frac = len(y) / max_iter
        c = 0.10 * (1.0 - frac) + 0.01 * frac          # 0.10 -> 0.01 across the run
        exploreRate = c * sigma

        best = np.max(y)                               # incumbent = best OBSERVED value
        x_next = Opt_Acquisition(GP_model, bounds, best, explore=exploreRate)
        x_next = x_next.round(decimals=5)              # NB: assumes ~O(1)-scale params

        params = {"ID": idx}
        for p, i in zip(bounds, range(len(bounds))):
            params[p] = x_next[0, i]

        val = safe_objective(obj_fn, params)
        if val is None:
            # Penalize: make a failed run look worse than anything seen, so the
            # GP/EI steer away from this region (low objective == high loss).
            val = float(np.min(y)) - 0.1 * (abs(float(np.min(y))) + 1.0)
            print("  -> evaluation failed; penalized with", val)

        X = np.vstack([X, x_next])
        y = np.vstack([y, val])
        idx_list = np.vstack([idx_list, idx])

        best_so_far = np.argmax(y)
        print("Best objective (=-loss): %.6g" % np.max(y))
        print("Best loss (if Objective returns -loss): %.6g" % (-np.max(y)))
        print("Best params:", X[best_so_far])

        # Update log (full rewrite each step), current length scales at the top.
        write_log(log_file, idx_list, X, y, bounds, length_scales=current_ls)

        idx += 1

    # After the loop: fit once more on ALL data and read off the ARD length
    # scales (one per parameter, in the column order of `bounds`). With an ARD
    # kernel `GP_model.kernel_.length_scale` is a vector; kernel_ (trailing
    # underscore) is the *optimized* kernel, kernel is the initial one.
    length_scales = None
    if len(y) >= 1:
        GP_model.fit(X, y.ravel())
        _ls = np.atleast_1d(GP_model.kernel_.length_scale)
        length_scales = [float(v) for v in _ls]
        print("Fitted ARD length scales (parameter : length_scale):")
        for _p, _v in zip(bounds, length_scales):
            print("  %-10s : %.6g" % (_p, _v))
        # Final rewrite so the log's header carries the final, full-data length
        # scales (also refreshes the header when resuming with nothing to do).
        write_log(log_file, idx_list, X, y, bounds, length_scales=length_scales)
    """

    # Hand the fitted length scales back to Julia as param => length_scale.
    ls = py"length_scales"
    isnothing(ls) && return nothing
    return OrderedDict(String(k) => ls[i] for (i, k) in enumerate(ks))
end
