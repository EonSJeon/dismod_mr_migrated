import numpy as np
import pymc as pm
import pytensor.tensor as at


def build_W_linear(knots: np.ndarray, ages: np.ndarray) -> np.ndarray: # Checked

    N, K = ages.size, knots.size
    W = np.zeros((N, K), dtype=float)
    idx = np.searchsorted(knots, ages, side="right")
    for i, (a, j_plus) in enumerate(zip(ages, idx)):
        j_minus = j_plus - 1
        if j_plus == K and np.isclose(a, knots[-1]):          # exactly last knot
            W[i, -1] = 1.0
        elif 0 < j_plus < K:                                  # interior interval
            lk, rk = knots[j_minus], knots[j_plus]
            W[i, j_minus] = (rk - a) / (rk - lk)
            W[i, j_plus]  = (a  - lk) / (rk - lk)
    return W


def spline(data_type: str) -> at.TensorVariable:
    ### Setting
    pm_model  = pm.modelcontext(None)
    sd        = pm_model.shared_data
    params_dt = sd["parameters"][data_type]

    # --- ages (must exist) ---
    if "age" not in pm_model.coords:
        raise ValueError("coords['age'] is missing. Register it upstream via pm_model.add_coord('age', ages, mutable=False).")
    ages = np.asarray(pm_model.coords["age"], dtype=float)

    # --- knots ---
    knot_dim = f"knot_{data_type}"
    if knot_dim in pm_model.coords:
        raise ValueError(f"coords['{knot_dim}'] already exists; spline() should be called only once per data_type.")

    if "age" not in pm_model.coords:
        raise ValueError("coords['age'] is missing. Register it upstream via pm_model.add_coord('age', ages, mutable=False).")
    ages      = np.asarray(pm_model.coords["age"], dtype=float)
    first_age = float(ages[0])
    last_age  = float(ages[-1])

    raw_knots = params_dt.get("knots", np.arange(first_age, last_age + 1, 5))
    knots = np.asarray(raw_knots, dtype=float).ravel()
    if knots.size == 0:
        raise ValueError("knots must contain at least one knot.")
    
    knots.sort()

    # --- duplicate check  ---
    vals, counts = np.unique(knots, return_counts=True)
    dups = vals[counts > 1]
    if dups.size > 0:
        raise ValueError(f"Duplicate knot value(s) found: {dups.tolist()}")

    # --- start constraint: cannot precede first age ---
    if knots[0] < first_age:
        raise ValueError(f"First knot ({knots[0]}) cannot be before first age ({first_age}).")

    # --- end constraint: cannot exceed last age ---
    if knots[-1] > last_age:
        raise ValueError(f"Last knot ({knots[-1]}) cannot exceed last age ({last_age}).")

    # --- ensure endpoints present (no re-check of spacing, same policy as tail) ---
    if not np.isclose(knots[0], first_age):
        knots = np.concatenate([[first_age], knots])
    if not np.isclose(knots[-1], last_age):
        knots = np.concatenate([knots, [last_age]])

    pm_model.add_coord(knot_dim, knots, mutable=False)

    # --- interpolation method ---
    method = params_dt.get("interpolation_method", "linear")
    if method != "linear":
        raise ValueError(f"Only linear splines supported, got {method!r}")

    # --- smoothness ---
    smooth_map = {"No Prior": None, "Slightly": 0.5, "Moderately": 0.05, "Very": 0.005}
    print(f"smooth_map: {smooth_map}")
    raw_smooth = params_dt.get("smoothness", "No Prior")
    print(f"raw_smooth: {raw_smooth}")
    if isinstance(raw_smooth, dict):
        amt = raw_smooth.get("amount", "No Prior")
        if isinstance(amt, str):
            if amt not in smooth_map:
                raise ValueError(f"Invalid smoothness amount '{amt}'.")
            smoothing = smooth_map[amt]
        elif isinstance(amt, (int, float)):
            smoothing = float(amt)
        else:
            raise TypeError("`amount` must be str|int|float")
    elif isinstance(raw_smooth, str):
        if raw_smooth not in smooth_map:
            raise ValueError(f"Invalid smoothness '{raw_smooth}'.")
        smoothing = smooth_map[raw_smooth]
    else:
        raise TypeError("`smoothness` must be dict or one of {'No Prior','Slightly','Moderately','Very'}")

    ### Main ###
    # --- design matrix (constant) ---
    W = at.constant(build_W_linear(knots, ages))

    # --- knot log-values & positive heights ---
    gamma_init = np.full(knots.size, -10.0)
    gamma     = pm.Normal(f"gamma_{data_type}", mu=0.0, sigma=10.0, dims=(knot_dim,), initval=gamma_init)
    exp_gamma = at.exp(gamma)

    # --- assemble mu(age) ---
    mu_age = pm.Deterministic(f"mu_age_{data_type}", at.dot(W, exp_gamma), dims=("age",))

    # --- rounded log-smoothing penalty (skip if No Prior) ---
    # --- smooth prior: match the original exactly (no division by total length) ---
    if smoothing is not None:
        # γ_min = log( (∑ exp(γ))/ (10*K) )
        gamma_min = at.log(at.sum(exp_gamma) / (10.0 * knots.size))
        clipped   = at.switch(gamma < gamma_min, gamma_min, gamma)

        # diffs = clipped[k] - clipped[k+1]
        diffs    = clipped[:-1] - clipped[1:]

        # denominator: ONLY the interval length Δa_k (no / total_length)
        intervals = knots[1:] - knots[:-1]
        denom_t   = at.constant(intervals)

        # S = ∑ (diffs^2 / Δa_k)
        S = at.sum((diffs**2) / denom_t)

        # same as original: -0.5 * S / σ^2
        pm.Potential(f"smooth_{data_type}", -0.5 * S / (smoothing**2))


    return mu_age


