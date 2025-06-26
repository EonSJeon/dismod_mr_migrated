import numpy as np
import pandas as pd
import pymc as pm
import warnings
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at   # ← import cumsum, etc.
import pytensor   # 추가된 import

# ------------------------------
# Age integrating models
# ------------------------------
def age_standardize_approx(
    name: str,
    age_weights: np.ndarray,
    mu_age: at.TensorVariable,
    age_start: np.ndarray,
    age_end: np.ndarray,
    ages: np.ndarray,
) -> dict[str, at.TensorVariable]:
    """
    Approximate interval average of mu_age over [age_start, age_end] with weights.
    Returns dict with 'mu_interval' deterministic.
    """
    assert pm.modelcontext(None) is not None, 'age_standardize_approx must be called within a PyMC model'

    # 1) cumulative weights (NumPy)
    cum_wt_np = np.cumsum(age_weights)

    # 2) clip and convert age_start/age_end to integer indices into 'ages'
    start_idx_np = (age_start.clip(ages[0], ages[-1]) - ages[0]).astype(int)
    end_idx_np   = (age_end.clip(ages[0], ages[-1])   - ages[0]).astype(int)

    # 3) build PyTensor constants for those indices and cumulative weights
    start_idx_tt = at.constant(start_idx_np)
    end_idx_tt   = at.constant(end_idx_np)
    cum_wt_tt    = at.constant(cum_wt_np)

    # 4) build cumulative weighted mu on the PyTensor side
    cum_mu = pm.Deterministic(
        f"cum_sum_mu_{name}",
        at.cumsum(mu_age * age_weights)
    )

    # 5) extract cum_mu at start and end indices via at.take
    cum_mu_start = at.take(cum_mu, start_idx_tt)
    cum_mu_end   = at.take(cum_mu, end_idx_tt)

    # 6) compute interval sum and interval total weight
    interval_sum  = cum_mu_end - cum_mu_start
    interval_wt   = at.take(cum_wt_tt, end_idx_tt) - at.take(cum_wt_tt, start_idx_tt)

    # 7) naive interval mean (PyTensor)
    vals = interval_sum / interval_wt

    # 8) handle zero-length intervals (where start_idx == end_idx)
    eq_mask       = at.eq(start_idx_tt, end_idx_tt)
    mu_at_start   = at.take(mu_age, start_idx_tt)
    vals_fixed    = at.switch(eq_mask, mu_at_start, vals)

    # 9) make the result a Deterministic
    mu_interval = pm.Deterministic(f"mu_interval_{name}", vals_fixed)

    return {'mu_interval': mu_interval}


#================================================================================

def age_integrate_approx(
    name: str,
    age_weights: List[str],
    mu_age: Any,
    age_start: Any,
    age_end: Any,
    ages: np.ndarray
) -> Dict[str, Any]:
    """
    Approximate interval average with per-interval semicolon-delimited weights.
    """
    # parse weights
    W = [np.array([1e-9 + float(w) for w in wi.split(';')][:-1]) for wi in age_weights]
    sumW = np.array([w.sum() for w in W])
    # indices
    start_idx = (age_start.__array__().clip(ages[0], ages[-1]) - ages[0]).astype(int)
    end_idx = (age_end.__array__().clip(ages[0], ages[-1]) - ages[0]).astype(int)
    # compute means
    def _int_means():
        N = len(W)
        mu = np.zeros(N)
        for i in range(N):
            mu[i] = np.dot(W[i], mu_age[start_idx[i]:end_idx[i]]) / sumW[i]
        return mu
    mu_interval = pm.Deterministic(f"mu_interval_{name}", _int_means())
    return { 'mu_interval': mu_interval }

def midpoint_approx(
    name: str,
    mu_age: Any,
    age_start: Any,
    age_end: Any,
    ages: np.ndarray
) -> Dict[str, Any]:
    """
    Approximate interval mean using midpoint value of mu_age.
    """
    mid = ((age_start + age_end) / 2.).astype(int)
    idx = np.clip(mid, ages[0], ages[-1]) - ages[0]
    mu_interval = pm.Deterministic(f"mu_interval_{name}", mu_age.take(idx))
    return { 'mu_interval': mu_interval }

def midpoint_covariate_approx(
    name: str,
    mu_age: Any,
    age_start: Any,
    age_end: Any,
    ages: np.ndarray,
    transform: Any = lambda x: x
) -> Dict[str, Any]:
    """
    Midpoint interval approx with linear covariate adjustment.
    Returns dict with 'mu_interval' and 'theta'.
    """
    # base covariate
    theta = pm.Normal(f"theta_{name}", mu=0.0, sigma=0.1, initval=0.0)
    # midpoints and widths
    mid = ((age_start + age_end) / 2.).astype(int)
    width = transform(age_end - age_start)
    idx = np.clip(mid, ages[0], ages[-1]) - ages[0]
    vals = mu_age.take(idx) + theta * width
    mu_interval = pm.Deterministic(f"mu_interval_{name}", vals)
    return { 'mu_interval': mu_interval, 'theta': theta }
