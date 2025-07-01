import numpy as np
import pymc as pm
import pytensor.tensor as at


def build_W_linear(knots: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """
    Return an (NxK) weight matrix W so that W @ y yields the piecewise-linear interpolation
    of knot-values y at each of the N query ages.
    """
    N, K = ages.size, knots.size
    W = np.zeros((N, K))
    idx = np.searchsorted(knots, ages, side="right")
    for i, (a, j_plus) in enumerate(zip(ages, idx)):
        j_minus = j_plus - 1
        # exactly last knot
        if j_plus == K and np.isclose(a, knots[-1]):
            W[i, -1] = 1.0
        # interior interval
        elif 0 < j_plus < K:
            lk, rk = knots[j_minus], knots[j_plus]
            W[i, j_minus] = (rk - a) / (rk - lk)
            W[i, j_plus]  = (a  - lk) / (rk - lk)
    return W


def spline() -> at.TensorVariable:
    """
    Inside a pm.Model, build a positive piecewise-linear function mu(age) from shared data,
    and optionally add a smoothness-penalty potential on its log-heights.
    """
    model  = pm.modelcontext(None)
    dt     = model.shared_data["data_type"]
    knots  = model.shared_data["knots"]
    ages   = model.shared_data["ages"]
    smooth = model.shared_data["smoothing"]
    method = model.shared_data["interpolation_method"]

    if method != "linear":
        raise ValueError(f"Only linear splines supported, got {method!r}")
    if not np.all(np.diff(knots) > 0):
        raise ValueError("Knots must be strictly increasing")

    W = at.constant(build_W_linear(knots, ages))
    
    model.add_coord("knot", knots)
    gamma   = pm.Normal(f"gamma_{dt}", mu=0, sigma=10, dims=("knot",))
    heights = at.exp(gamma)

    model.add_coord("age", ages)
    mu_age = pm.Deterministic(name=f"mu_age_{dt}", var=at.dot(W, heights), dims=("age",))

    if smooth and np.isfinite(smooth):
        gamma_min = at.log(at.sum(heights) / 10 / knots.size)
        clipped   = at.switch(gamma < gamma_min, gamma_min, gamma)
        diffs     = clipped[:-1] - clipped[1:]
        inv_denom = 1 / ((knots[1:] - knots[:-1]) * (knots[-1] - knots[0]))
        penalty   = 0.5 * at.sum(diffs**2 * inv_denom) / smooth**2
        pm.Potential(f"smooth_{dt}", -penalty)

    return mu_age