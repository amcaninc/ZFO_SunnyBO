

function Fit(Objective, interval, max_iter; start_point=nothing)
    DIR = @__DIR__
    @pyinclude(DIR*"/Bopt.py")
    
    # Build the initial point: use start_point if provided, else midpoint
    if isnothing(start_point)
        initial_x = nothing  # signal to Python to use midpoint
    else
        initial_x = [start_point[p] for p in keys(interval)]
    end
    
    py"""
    bounds = $interval
    max_iter = $max_iter
    initial_x = $initial_x  # None or a list of values

    $SafetyChecks(bounds)

    kernel = Matern(
        length_scale=1.0, 
        length_scale_bounds=(1e-03, 1e3), 
        nu=2.5
    )
    GP_model = GaussianProcessRegressor(
        kernel=kernel, 
        normalize_y=False, 
        optimizer='fmin_l_bfgs_b',  
        n_restarts_optimizer=30  
    )
    
    X, y, idx_list = Restart(bounds)
    start = int(idx_list[-1][0]) + 1

    if start == 1:
        print("Initial Run")
        idx_list = np.array([start])
        if initial_x is None:
            X = np.vstack([np.mean(bounds[p]) for p in bounds]).T  # original midpoint
        else:
            X = np.array(initial_x).reshape(1, -1)                 # your start point
        params = {"ID": 1}
        for p, i in zip(bounds, range(len(bounds))):
            params[p] = X[0, i]
        y = np.array([$Objective(params)])
        start = start + 1

    for idx in range(start, max_iter):
        print("Bayesian Opt Step :: %i"%idx)
        # Fit GP
        GP_model.fit(X, y)
    
        # Enforce alteration of high exploration and exploitation
        if idx < 200:
            exploreRate = 0.5
        elif idx % 4 == 0 : 
            exploreRate = 0.25 
        else : 
            exploreRate = 0.0
        
        # Evaluate Objective
        x_next = Opt_Acquisition(X, GP_model, bounds=bounds, explore=exploreRate)
        x_next = x_next.round(decimals=5, out=None)
        params ={"ID":idx}
        for p,i in zip(bounds, range(len(bounds))):
            params[p] = x_next[0,i]
        X = np.vstack([X, x_next])
        y = np.vstack([y, $Objective(params) ])
        idx_list = np.vstack([idx_list, idx])
        
        best_so_far = np.argmax(y)
        print("Best Loss ", np.max(y), '\n', "Params = ", X[best_so_far])
        
        # Update Log
        data = np.hstack((idx_list, X, y))
        header = [p for p in bounds]
        header.append("Obj")
        header.insert(0, "ID")
        df = pd.DataFrame(data, columns=header,)
        df["ID"] = df["ID"].astype(int)
        df.to_csv("Bopt_Log.csv", sep='\t')
    """ 
end
