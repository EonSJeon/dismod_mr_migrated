import numpy as np
import pymc as pm
import pytensor.tensor as at


def similar(
    child_curve,
    parent_curve,
    sigma_parent,      # 선형 스케일 표준편차(연령별 벡터/스칼라 모두 허용)
    eps=1e-9,
    penalty_name="",
    sigma_diff_log=0.0 # (선택) 구조적 차이 허용, 문헌식과 동일하려면 0으로 두세요
):
    pm_model = pm.modelcontext(None)
    label    = pm_model.shared_data.get("data_type", "")

    ylog = at.log(child_curve  + eps)
    mlog = at.log(parent_curve + eps)

    sigma_log = (at.as_tensor_variable(sigma_parent) + eps) / (parent_curve + eps)
    if sigma_diff_log and float(sigma_diff_log) > 0:
        sigma_log = pm.math.sqrt(sigma_log**2 + float(sigma_diff_log)**2)

    # 안전 바닥(0 분산 방지)
    sigma_log = pm.math.clip(sigma_log, 1e-12, np.inf)

    lp = pm.logp(pm.Normal.dist(mu=mlog, sigma=sigma_log), ylog)
    return pm.Potential(f"parent_similarity_{label}{penalty_name}", at.sum(lp))


def level_constraints(data_type: str, unconstrained_mu_age: at.TensorVariable):
    """
    Clip `unconstrained_mu_age` to a fixed level outside [age_before, age_after],
    and softly penalize deviation from the original *within* that interval.

    Requires coords to be pre-registered:
      - coords['age'] : age grid (length N)
    """
    pm_model = pm.modelcontext(None)
    sd       = pm_model.shared_data
    params_of_data_type   = sd["parameters"][data_type]

    if ("level_value" not in params_of_data_type) or ("level_bounds" not in params_of_data_type):
        return unconstrained_mu_age

    # ---- guards ----
    if "age" not in pm_model.coords:
        raise ValueError("coords['age'] is missing. Register it upstream.")
    ages = np.asarray(pm_model.coords["age"], dtype=float)
    if ages.ndim != 1 or ages.size == 0:
        raise ValueError("coords['age'] must be a 1D non-empty array.")

    # ---- config ----
    lv = params_of_data_type["level_value"]
    lb = float(params_of_data_type["level_bounds"]["lower"])
    ub = float(params_of_data_type["level_bounds"]["upper"])
    if not (np.isfinite(lb) and np.isfinite(ub) and lb < ub):
        raise ValueError(f"Invalid level_bounds: lower={lb}, upper={ub}")

    level_value = float(lv["value"])
    age_before  = float(lv["age_before"])
    age_after   = float(lv["age_after"])

    if age_after < age_before:
        raise ValueError(f"'age_after' ({age_after}) must be >= 'age_before' ({age_before}).")

    # ---- map ages → index range using searchsorted (grid spacing may be != 1) ----
    i_start = int(np.clip(np.searchsorted(ages, age_before, side="left"),  0, ages.size - 1))
    i_end   = int(np.clip(np.searchsorted(ages, age_after,  side="right") - 1, 0, ages.size - 1))

    idx = at.arange(ages.size)
    val = at.as_tensor_variable(level_value)

    ### Clipping ###
    # ---- hard clipping age before and after ----
    mid   = at.switch((idx >= i_start) & (idx <= i_end), unconstrained_mu_age, val)
    clipped = at.switch(idx < i_start, val, at.switch(idx > i_end, val, mid))

    constrained = pm.Deterministic(
        f"constrained_mu_age_{data_type}",
        at.clip(clipped, lb, ub),
        dims=("age",),
    )

    # ---- soft similarity penalty ONLY within [i_start, i_end] ----
    if i_end >= i_start:
        child_slice  = constrained[i_start : i_end + 1]
        parent_slice = unconstrained_mu_age[i_start : i_end + 1]
        similar(
            child_curve     = child_slice,
            parent_curve    = parent_slice,
            sigma_parent    = 0.0,     # parent treated as fixed target
            sigma_diff_log  = 0.01,    # small allowance in log-space
            eps             = 1e-6,
            penalty_name    = "_level_constraints",
        )

    return constrained


def derivative_constraints(data_type: str, mu_age: at.TensorVariable):
    pm_model = pm.modelcontext(None)
    sd       = pm_model.shared_data

    # --- params 로드 ---
    params_of_data_type = sd["parameters"][data_type]
    inc = params_of_data_type.get("increasing")
    dec = params_of_data_type.get("decreasing")
    if not inc and not dec:
        return None  
    
    # --- coords['age'] 필수 ---
    if "age" not in pm_model.coords:
        raise ValueError("coords['age'] is missing. Register it upstream.")
    ages = np.asarray(pm_model.coords["age"], dtype=float)
    if ages.ndim != 1 or ages.size < 2:
        raise ValueError("coords['age'] must be a 1D array with length >= 2.")
    if np.any(np.diff(ages) <= 0):
        raise ValueError("coords['age'] must be strictly increasing.")


    # --- helper: [a_start, a_end] → diff 인덱스 구간 [i0, i1_excl] (on diff(mu) of length N-1) ---
    def _diff_span(a_start, a_end, grid):
        if a_end < a_start:
            raise ValueError(f"age_end ({a_end}) must be >= age_start ({a_start}).")
        N = grid.size
        # age 인덱스 범위(포함)
        i_start_age = int(np.searchsorted(grid, float(a_start), side="left"))
        i_end_age   = int(np.searchsorted(grid, float(a_end), side="right") - 1)
        i_start_age = max(0, min(i_start_age, N - 1))
        i_end_age   = max(0, min(i_end_age,   N - 1))
        # diff 인덱스는 0..N-2, 각 항은 (age[i+1]-age[i])에 대응
        i0 = i_start_age
        i1_excl = min(i_end_age, N - 2) + 1  # 포함 끝 → 슬라이스 끝+1
        if i1_excl <= i0:
            return None
        return (i0, i1_excl)

    # --- overlap 검사---
    if inc and dec:
        a_inc_start, a_inc_end = float(inc["age_start"]), float(inc["age_end"])
        a_dec_start, a_dec_end = float(dec["age_start"]), float(dec["age_end"])
        if max(a_inc_start, a_dec_start) <= min(a_inc_end, a_dec_end):
            raise ValueError(
                f"Increasing [{a_inc_start}, {a_inc_end}] overlaps with decreasing [{a_dec_start}, {a_dec_end}]."
            )

    # --- diff와 위반량 계산 ---
    diff = at.diff(mu_age)  # shape: (len(ages)-1,)
    terms = []

    if inc:
        span = _diff_span(inc["age_start"], inc["age_end"], ages)
        if span is not None:
            s, e = span
            # 증가 구간에서 음의 기울기(<=0) 벌점
            inc_viol = at.sum(at.clip(diff[s:e], -np.inf, 0.0))
            terms.append(inc_viol**2)

    if dec:
        span = _diff_span(dec["age_start"], dec["age_end"], ages)
        if span is not None:
            s, e = span
            # 감소 구간에서 양의 기울기(>=0) 벌점
            dec_viol = at.sum(at.clip(diff[s:e], 0.0, np.inf))
            terms.append(dec_viol**2)

    if not terms:
        return None

    penalty = terms[0] if len(terms) == 1 else at.sum(at.stack(terms))
    logp    = -1e12 * penalty  

    return pm.Potential(
        name=f"mu_age_derivative_potential_{data_type}",
        var=logp,
    )


def covariate_level_constraints(data_type, mu_age) -> at.TensorVariable:
    """
    Enforce level‐bounds on the covariate‐adjusted rate curve.
    If bounds['lower'] == 0, we skip the lower‐bound term entirely.
    """
    # --------------------------- 1) Initialize PyMC model ---------------------------   
    pm_model = pm.modelcontext(None)  # reforged_mr/model/priors/covariate_level_constraints()
    sd = pm_model.shared_data
    params = sd["parameters"]
    params_dt = params[data_type]

    # --------------------------- 2) Extract shared data -----------------------------   
    region_id_graph = sd["region_id_graph"]
    global_id = sd["global_id"]
    max_depth = sd["max_depth"]
    lvl    = params_dt.get('level_value')
    bounds = params_dt.get('level_bounds')

    U           = sd[f'U_{data_type}']
    alpha       = sd[f'alpha_{data_type}']

    X_centering = sd[f'X_centering_{data_type}']
    X_scaling   = sd[f'X_scaling_{data_type}']
    beta        = sd[f'beta_{data_type}']
    

    # Exit if no level info
    if not lvl or not bounds:
        return {}

    # --------------------------- 3) Compute sex covariate range (after centering) --- 
    sex_idx    = list(X_centering.index).index('x_sex')
    X_sex_max  =  0.5 - X_centering['x_sex'] / X_scaling['x_sex']
    X_sex_min  = -0.5 - X_centering['x_sex'] / X_scaling['x_sex']

    # --------------------------- 4) Build “layers” of U‐masks -----------------------
    layers: list[np.ndarray] = []

    nodes = [global_id]
    for _ in range(3):
        nodes = [c for n in nodes for c in region_id_graph.successors(n)]
        mask  = np.array([col in nodes for col in U.columns], dtype=bool)
        if mask.any():
            layers.append(mask)

    # --------------------------- 5) Prepare log‐bounds ------------------------------
    lower_val = bounds['lower']
    if lower_val <= 0:
        low = None 
    else:
        low = np.log(lower_val)

    high = np.log(bounds['upper'])  # high bound always exists

    # --------------------------- 6) Build potential inside PyMC ---------------------
    mu        = mu_age
    log_vals  = at.log(mu)           # log(mu) at each age
    log_max   = at.max(log_vals)     # base‐curve extrema
    log_min   = at.min(log_vals)

    # (a) Add random‐effect contributions
    alpha_list = [] if alpha is None else alpha
    if len(alpha_list) > 0:
        alphas = at.stack(alpha_list)  # shape=(n_re,)
        for m in layers:
            m_idx = np.where(m)[0]
            if m_idx.size > 0:
                sub_alphas = alphas[m_idx]
                log_max += at.max(sub_alphas)
                log_min += at.min(sub_alphas)

    # (b) Add sex fixed‐effect
    b_sex = beta[sex_idx]
    try:
        log_max += X_sex_max * b_sex
        log_min += X_sex_min * b_sex
    except (TypeError, AttributeError):  # numeric case
        log_max += X_sex_max * float(b_sex)
        log_min += X_sex_min * float(b_sex)

    # (c) Compute “below‐lower” violation (if applicable)
    if low is None:
        v_low = at.constant(0.0)
    else:
        v_low = at.minimum(0, log_min - low)  # negative if below bound

    # (d) Compute “above‐upper” violation
    v_high = at.maximum(0, log_max - high)    # positive if above bound

    # --------------------------- 7) Apply tight Normal(0, 1e-6) penalty --------------
    sigma       = 1e-6
    stacked_v   = at.stack([v_low, v_high])
    norm_dist   = pm.Normal.dist(mu=0.0, sigma=sigma)
    logp_vals   = pm.logp(norm_dist, stacked_v)
    logp_sum    = at.sum(logp_vals)

    # --------------------------- 8) Register as Potential ---------------------------
    covariate_constraint = pm.Potential(
        f"covariate_constraint_{data_type}",
        var=logp_sum
    )

    return covariate_constraint
