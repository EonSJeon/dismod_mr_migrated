import numpy as np
import pymc as pm
import pytensor.tensor as at


def similar(child_curve, parent_curve, sigma_parent, sigma_diff, eps=1e-9, penalty_name=""):
    """
    Softly shrink child_curve toward parent_curve on the log scale.
    """
    model      = pm.modelcontext(None) # reforged_mr - similar()
    label      = model.shared_data["data_type"]
    # determine precision τ
    if hasattr(parent_curve, "distribution"):
        tau = 1/(sigma_parent**2 + sigma_diff**2)
    else:
        tau = 1/(((sigma_parent + eps)/(parent_curve + eps))**2 + sigma_diff**2)
    # log‐values, clipped to avoid log(0)
    log_child  = at.log(pm.math.clip(child_curve,  eps, np.inf))
    log_parent = at.log(pm.math.clip(parent_curve, eps, np.inf))
    # elementwise log-density, then sum into one potential
    lp = pm.logp(
        pm.Normal.dist(mu=log_parent, sigma=1/pm.math.sqrt(tau)),
        log_child
    )
    pm.Potential(f"parent_similarity_{label}{penalty_name}", pm.math.sum(lp))


def level_constraints(unconstrained_mu_age: at.TensorVariable):
    """
    Hard-clip unconstrained_mu_age outside [before, after] to a fixed value,
    then softly penalize deviation from the original in between.
    """
    model      = pm.modelcontext(None) # reforged_mr - level_constraints()
    label      = model.shared_data["data_type"]
    ages       = model.shared_data["ages"]
    params     = model.shared_data["params_of_data_type"]
    # exit if no level constraints provided
    if not ("level_value" in params and "level_bounds" in params):
        return unconstrained_mu_age, unconstrained_mu_age, None

    lv   = params["level_value"]
    lb   = params["level_bounds"]["lower"]
    ub   = params["level_bounds"]["upper"]
    # map ages to indices
    start = int(np.clip(lv["age_before"] - ages[0], 0, ages.size))
    end   = int(np.clip(lv["age_after"]  - ages[0], 0, ages.size))
    idx   = at.arange(ages.size)
    val   = float(lv["value"])

    # piecewise: val before start, raw in [start,end], val after end
    clipped = at.switch(idx < start, val,
                at.switch(idx > end, val, unconstrained_mu_age))
    constrained_mu_age = at.clip(clipped, lb, ub)
    pm.Deterministic(f"constrained_mu_age_{label}", constrained_mu_age)

    # add similarity potential back to the raw curve
    sim = similar(
        child_curve      = constrained_mu_age,
        parent_curve     = unconstrained_mu_age,
        sigma_parent     = 0.0,
        sigma_diff       = 0.01,
        eps              = 1e-6,
        penalty_name     = "_level_constraints"
    )
    return constrained_mu_age


def derivative_constraints(mu_age: at.TensorVariable):
    """
    Enforce monotonicity by heavily penalizing negative (or positive)
    finite-differences on specified age-ranges.
    """
    model      = pm.modelcontext(None) # reforged_mr - derivative_constraints()
    ages       = model.shared_data["ages"]
    params     = model.shared_data["params_of_data_type"]
    inc, dec   = params.get("increasing"), params.get("decreasing")
    if not (inc and dec):
        return {}

    # helper to turn an age into a safe diff-index
    def to_idx(age_val):
        idx = age_val - ages[0]
        return int(np.clip(idx, 0, len(ages) - 1))

    i0, i1 = to_idx(inc["age_start"]), to_idx(inc["age_end"])
    d0, d1 = to_idx(dec["age_start"]), to_idx(dec["age_end"])

    diff = at.diff(mu_age)
    inc_viol = at.sum(at.clip(diff[i0:i1], -np.inf,   0.0))
    dec_viol = at.sum(at.clip(diff[d0:d1],   0.0, np.inf))

    penalty = inc_viol**2 + dec_viol**2
    logp    = -1e12 * penalty

    pm.Potential(
        name=f"mu_age_derivative_potential_{model.shared_data['data_type']}",
        var=logp
    )


def covariate_level_constraints(X_shift, beta, U, alpha, mu_age) -> at.TensorVariable:
    """
    Enforce level‐bounds on the covariate‐adjusted rate curve.
    If bounds['lower'] == 0, we skip the lower‐bound term entirely.
    """

    # --------------------------- 1) initialize pm_model ---------------------------   
    pm_model = pm.modelcontext(None) # at reforged_mr/model/priors/covariate_level_constraints()


    # --------------------------- 2) extract shared data ---------------------------   
    data_type = pm_model.shared_data["data_type"]
    region_id_graph = pm_model.shared_data["region_id_graph"]
    params = pm_model.shared_data["params_of_data_type"]
    lvl    = params.get('level_value')
    bounds = params.get('level_bounds')
    if not lvl or not bounds:
        # nothing to do if no level_value or no level_bounds
        return {}

    # 2) Compute sex covariate range (after centering)
    sex_idx = list(X_shift.index).index('x_sex')
    X_sex_max = 0.5 - X_shift['x_sex']
    X_sex_min = -0.5 - X_shift['x_sex']

    # 3) Build “layers” of U‐masks for the random‐effects hierarchy
    layers: list[np.ndarray] = []

    global_id = 1 # TODO: make it a parameter later 
    nodes = [global_id]
    for _ in range(3):
        nodes = [c for n in nodes for c in region_id_graph.successors(n)]
        mask = np.array([col in nodes for col in U.columns], dtype=bool)
        if mask.any():
            layers.append(mask)

    # 4) Prepare log‐bounds.  If lower==0, skip that term:
    lower_val = bounds['lower']
    if lower_val <= 0:
        # Skip any “below‐bound” penalty
        low = None
    else:
        low = np.log(lower_val)

    # high bound is always log(upper)
    high = np.log(bounds['upper'])

    # 5) Inside the PyMC model, build the potential:
    mu = mu_age  # (TensorVariable, shape=(len(ages),))

    # (a) log(mu) at each age
    log_vals = at.log(mu)

    # (b) base‐curve extrema
    log_max = at.max(log_vals)
    log_min = at.min(log_vals)

    # (c) add random‐effect contributions, if any
    if alpha is None:
        alpha_list = []
    else:
        alpha_list = alpha

    if len(alpha_list) > 0:
        alphas = at.stack(alpha_list)  # shape=(n_re,)
        for m in layers:
            m_idx = np.where(m)[0]
            if m_idx.size > 0:
                sub_alphas = alphas[m_idx]
                log_max = log_max + at.max(sub_alphas)
                log_min = log_min + at.min(sub_alphas)

    # (d) add sex fixed‐effect
    b_sex = beta[sex_idx]
    try:
        # if b_sex is a TensorVariable
        log_max = log_max + X_sex_max * b_sex
        log_min = log_min + X_sex_min * b_sex
    except (TypeError, AttributeError):
        # b_sex was numeric
        log_max = log_max + X_sex_max * float(b_sex)
        log_min = log_min + X_sex_min * float(b_sex)

    # (e) compute “below‐lower” violation only if low is not None
    if low is None:
        v_low = at.constant(0.0)
    else:
        v_low = at.minimum(0, log_min - low)
        # (if log_min < low, then log_min–low < 0, so v_low<0; else v_low=0)

    # (f) always compute “above‐upper” violation
    v_high = at.maximum(0, log_max - high)
    # (if log_max > high, then log_max–high>0, so v_high>0; else v_high=0)

    # (g) put them through a very‐tight Normal(0, 1e‐6) penalty
    sigma = 1e-6
    stacked_v = at.stack([v_low, v_high])
    norm_dist = pm.Normal.dist(mu=0.0, sigma=sigma)
    logp_vals = pm.logp(norm_dist, stacked_v)
    logp_sum  = at.sum(logp_vals)

    # (h) register as a single Potential
    covariate_constraint = pm.Potential(f"covariate_constraint_{data_type}", var=logp_sum)
    return covariate_constraint