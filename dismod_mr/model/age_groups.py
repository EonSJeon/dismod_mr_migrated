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
) -> Dict[str, Any]:
    """
    Approximate interval average of mu_age over [age_start, age_end] with weights.
    Returns dict with 'mu_interval' deterministic.
    """
    assert pm.modelcontext(None) is not None, 'age_standardize_approx must be called within a PyMC model'
    # cumulative weights
    cum_wt = np.cumsum(age_weights)
    # indices
    start_idx = (age_start.__array__().clip(ages[0], ages[-1]) - ages[0]).astype(int)
    end_idx = (age_end.__array__().clip(ages[0], ages[-1]) - ages[0]).astype(int)
    # cumulative weighted mu (use pytensor's cumsum, not pm.math)

    cum_mu = pm.Deterministic(
        f"cum_sum_mu_{name}",
        at.cumsum(mu_age * age_weights)
    )
    # compute interval means
    vals = (cum_mu[end_idx] - cum_mu[start_idx]) / (cum_wt[end_idx] - cum_wt[start_idx])
    # correct zero-length intervals
    eq = start_idx == end_idx
    if np.any(eq):
        vals = vals.copy()
        vals[eq] = mu_age[start_idx[eq]]
    mu_interval = pm.Deterministic(f"mu_interval_{name}", vals)

    return { 'mu_interval': mu_interval }


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
