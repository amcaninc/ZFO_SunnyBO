""" Basic Bayesian Optimization : Designed for PyCall.jl

    Hard-coded for arbitrary parameters and a noise-free scalar objective.

    Using Expected Improvement and a 5/2-Matern Kernel.

    SIGN CONVENTION: this maximizes the objective, but main file returns negative loss so this ends up minimizing
"""
import os, sys
import os.path
import math
import numpy as np
import scipy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from warnings import catch_warnings
from warnings import simplefilter
import warnings
warnings.filterwarnings(action='once')
import pandas as pd


def Restart(interval, log_file="Bopt_Log.csv"):
    """ Restart Bayesian Optimization from the same directory if possible.

    Args:
        interval (dict): Set of intervals/bounds for all parameters.
        log_file (str): CSV log to resume from. Defaults to "Bopt_Log.csv" so
            callers that don't tag their run keep the historical behavior.

    Returns:
        x_set   : Set of evaluated free parameters corresponding to `interval`.
        y_set   : Set of evaluated objective values (negative of the loss).
        idx_set : Set of indices corresponding to x_set and y_set.
    """
    if os.path.isfile(log_file):
        df_read = pd.read_csv(log_file, sep="\t", index_col=0, comment='#')

        # Guard against silently resuming a log that belongs to a DIFFERENT fit
        logged_params = list(df_read.columns[1:-1])
        expected_params = list(interval)
        if logged_params != expected_params:
            raise RuntimeError(
                "%s does not match the current bounds.\n"
                "  logged parameters : %s\n"
                "  current bounds    : %s\n"
                "This usually means a stale log from a previous run is present. "
                "Delete or rename %s to start fresh, or run in a clean "
                "directory." % (log_file, logged_params, expected_params, log_file)
            )

        D = df_read.to_numpy()
        idx_set = np.vstack([i for i in D[:, 0]])
        y_set = np.vstack([i for i in D[:, -1]])
        x_set = D[:, 1:-1]
    else:
        x_set = np.array([])
        y_set = np.array([])
        idx_set = np.array([[0.0]])   # sentinel: last ID == 0 means "fresh start"
    return x_set, y_set, idx_set


def write_log(log_file, idx_list, X, y, bounds, length_scales=None):
    """ Write the BO log to `log_file` as a TSV table, prefixed with the current
    ARD length scales as comment

    Args:
        log_file (str): destination TSV path.
        idx_list (array): column of evaluation IDs, shape (n, 1).
        X (array): evaluated parameters, shape (n, n_params).
        y (array): objective values, shape (n, 1).
        bounds (dict): parameter bounds; gives the column names / order.
        length_scales (list|None): current ARD length scales, one per parameter
            in `bounds` order. None -> not yet available (before the first GP fit).
    """
    data = np.hstack((idx_list, X, y))
    header = ["ID"] + [p for p in bounds] + ["Obj"]
    df = pd.DataFrame(data, columns=header)
    df["ID"] = df["ID"].astype(int)
    with open(log_file, "w", newline="") as f:
        f.write("# ARD length scales (current GP fit) -- larger = less sensitive\n")
        if length_scales is None:
            f.write("#   (not yet available: GP not fit until the first BO step)\n")
        else:
            for p, ls in zip(bounds, length_scales):
                f.write("#   %-10s : %.6g\n" % (p, ls))
        df.to_csv(f, sep='\t')


def safe_objective(obj_fn, params):
    """ Evaluate the objective and turn any failure into a clear `None`.

    Catches exceptions and non-finite (NaN/Inf) returns so that a single bad
    parameter set cannot crash the run or poison the GP fit.

    Args:
        obj_fn   : the (Julia) objective, callable as obj_fn(params).
        params (dict): {"ID": int, param_name: value, ...}

    Returns:
        float on success, or None on any failure.
    """
    try:
        v = obj_fn(params)
    except Exception as e:
        print("  objective raised an exception:", e)
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        print("  objective returned a non-numeric value:", repr(v))
        return None
    if not np.isfinite(v):
        print("  objective returned a non-finite value:", v)
        return None
    return v


def surrogate(model, X):
    """ Predict mean and std with the GP, suppressing warnings *locally only*.

    NOTE: the original used a process-wide warnings.filterwarnings("ignore"),
    which permanently silenced every warning (including GP convergence
    warnings). Here the suppression is scoped to this one call.

    Args:
        model (GP): trained surrogate GP model.
        X (array): parameters to evaluate.

    Returns:
        (mean, std): GP predictions.
    """
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std=True)


def Expected_Improvement(XS, model, best, explore):
    """ Expected Improvement acquisition score (for maximization).

    Args:
        XS (array): trial parameters to score, shape (n_candidates, n_params).
        model (GP): GP fit to the data so far.
        best (float): incumbent = best OBSERVED objective value, np.max(y).
        explore (float): exploration parameter (in objective units).

    Returns:
        EI (array): Expected Improvement per candidate, shape (n_candidates,).
    """
    mu, std = surrogate(model, XS)
    I = mu - best - explore
    Z = np.divide(I, std + 1e-8)
    EI = I * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
    EI[std < 1e-12] = 0.0          # no uncertainty -> no expected improvement
    return EI


def Opt_Acquisition(model, bounds, best, explore=0.0, n_candidates=2000):
    """ Find the next point to evaluate by maximizing EI over a candidate cloud.

    Draws a fixed large number of uniformly random candidates (independent of
    dimensionality) and returns the one with the highest EI.

    Args:
        model (GP): surrogate model.
        bounds (dict): {param_name: (low, high), ...}.
        best (float): incumbent objective value (np.max(y)).
        explore (float): exploration parameter, in objective units.
        n_candidates (int): size of the random candidate cloud.

    Returns:
        (array): the proposed next parameter set, shape (1, n_params).
    """
    cols = [np.random.uniform(bounds[b][0], bounds[b][1], size=n_candidates) for b in bounds]
    XR = np.array(cols).T                       # (n_candidates, n_params)
    EIScores = Expected_Improvement(XR, model, best, explore=explore)
    return np.array([XR[np.argmax(EIScores)]])
