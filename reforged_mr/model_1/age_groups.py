import numpy as np
import pandas as pd
import pymc as pm
import warnings
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at   # ← import cumsum, etc.


def age_standardize_approx(mu_age: at.TensorVariable, use_lb_data: bool = False) -> at.TensorVariable:
    """
    Approximate the interval average of mu_age over [age_start, age_end] using precomputed age_weights.
    """
    model = pm.modelcontext(None)
    dt    = model.shared_data["data_type"]
    ages  = model.shared_data["ages"]
    w     = model.shared_data["age_weights"]
    df    = model.shared_data["lb_data"] if use_lb_data else model.shared_data["data"]

    # align weight vector to the age grid
    if w.size != ages.size:
        w = w[:ages.size] if w.size > ages.size else np.pad(w, (0, ages.size - w.size), constant_values=0)

    # compute integer indices into the age grid
    start_idx_np = (df["age_start"].clip(ages[0], ages[-1]) - ages[0]).astype(int)
    end_idx_np   = (df["age_end"]  .clip(ages[0], ages[-1]) - ages[0]).astype(int)
    start_idx = at.constant(start_idx_np)
    end_idx   = at.constant(end_idx_np)

    # cumulative sums for numerator and denominator
    cum_w  = at.constant(np.cumsum(w))
    cum_mu = pm.Deterministic(f"cum_sum_mu_{dt}", at.cumsum(mu_age * w))

    # extract interval sums and weights
    interval_sum = at.take(cum_mu, end_idx) - at.take(cum_mu, start_idx)
    interval_wt  = at.take(cum_w,  end_idx) - at.take(cum_w,  start_idx)

    # compute weighted average, fallback to point value if start==end
    avg = at.switch(
        at.eq(start_idx, end_idx),
        at.take(mu_age, start_idx),
        interval_sum / interval_wt
    )

    mu_interval = pm.Deterministic(name=f"mu_interval_{dt}", var=avg)
    return mu_interval


#================================================================================

def age_integrate_approx(
    data_type: str,
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
    mu_interval = pm.Deterministic(f"mu_interval_{data_type}", _int_means())
    return { 'mu_interval': mu_interval }

def midpoint_approx(
    data_type: str,
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
    mu_interval = pm.Deterministic(f"mu_interval_{data_type}", mu_age.take(idx))
    return { 'mu_interval': mu_interval }

def midpoint_covariate_approx(
    data_type: str,
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
    theta = pm.Normal(f"theta_{data_type}", mu=0.0, sigma=0.1, initval=0.0)
    # midpoints and widths
    mid = ((age_start + age_end) / 2.).astype(int)
    width = transform(age_end - age_start)
    idx = np.clip(mid, ages[0], ages[-1]) - ages[0]
    vals = mu_age.take(idx) + theta * width
    mu_interval = pm.Deterministic(f"mu_interval_{data_type}", vals)
    return { 'mu_interval': mu_interval, 'theta': theta }
