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
    pm_model = pm.modelcontext(None)
    sd       = pm_model.shared_data
    params   = sd["parameters"]
    params_dt   = params[data_type]

    if ("level_value" not in params_dt) or ("level_bounds" not in params_dt):
        return unconstrained_mu_age

    # ---- guards ----
    if "age" not in pm_model.coords:
        raise ValueError("coords['age'] is missing. Register it upstream.")
    ages = np.asarray(pm_model.coords["age"], dtype=float)
    if ages.ndim != 1 or ages.size == 0:
        raise ValueError("coords['age'] must be a 1D non-empty array.")

    # ---- config ----
    lv = params_dt["level_value"]
    lb = float(params_dt["level_bounds"]["lower"])
    ub = float(params_dt["level_bounds"]["upper"])
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


    # ---- soft similarity penalty between constrained and unconstrained ----

    similar(
        child_curve     = constrained,
        parent_curve    = unconstrained_mu_age,
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
    params = sd["parameters"]
    params_dt = params[data_type]
    inc = params_dt.get("increasing")
    print(f"inc: {inc}")
    dec = params_dt.get("decreasing")
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
            raise ValueError(...)
        N = grid.size
        i_start_age = int(np.searchsorted(grid, float(a_start), side="left"))
        i_end_age   = int(np.searchsorted(grid, float(a_end),   side="right") - 1)
        i_start_age = max(0, min(i_start_age, N - 1))
        i_end_age   = max(0, min(i_end_age,   N - 1))

        # diff 인덱스 j는 (age[j], age[j+1]) 쌍을 뜻함.
        # 문헌: a = a_s..a_e-1 만 포함 → 슬라이스 끝은 index(a_e)
        i0 = i_start_age
        i1_excl = min(i_end_age, N - 1)     # ★ 여기서 +1 하지 않음
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


def covariate_level_constraints(data_type, mu_age):
    pm_model = pm.modelcontext(None)
    sd       = pm_model.shared_data
    params_dt = sd["parameters"][data_type]

    if not params_dt.get('include_covariates', True):
        raise ValueError("include_covariates is False. This function is only for covariates.")

    bounds = params_dt.get("level_bounds")
    if not bounds or ("upper" not in bounds):
        return None

    G      = sd["region_id_graph"]
    reference_area_id  = sd['reference_area_id']
    ref_level          = int(G.nodes[reference_area_id]['level'])
    max_level          = int(sd['max_depth']) - 1

    U           = sd[f"U_{data_type}"]
    X_centering = sd[f"X_centering_{data_type}"]
    X_scaling   = sd[f"X_scaling_{data_type}"]
    beta    = sd[f"beta_{data_type}"]          # vector TensorVariable (Deterministic)

    # ---- coords (필수) ----
    re_dim  = f"re_loc_id_{data_type}"
    fe_dim  = f"fe_eff_name_{data_type}"
    if re_dim not in pm_model.coords:
        raise ValueError(f"coord '{re_dim}' is missing. Register it when building U.")
    if fe_dim not in pm_model.coords:
        raise ValueError(f"coord '{fe_dim}' is missing. Register it when building X/beta.")

    # alpha 벡터 (Deterministic)
    alpha = pm_model.named_vars[f"alpha_{data_type}"]

    # 성별 FE 범위 (센터링/스케일링 반영)
    b_sex = None
    
    if "x_sex" in X_centering.index and "x_sex" in X_scaling.index:
        z_sex_max = ( 0.5 - X_centering["x_sex"]) / X_scaling["x_sex"]
        z_sex_min = (-0.5 - X_centering["x_sex"]) / X_scaling["x_sex"]
        try:
            sex_idx = list(X_centering.index).index("x_sex")
            b_sex   = beta[sex_idx]   # Subtensor OK
        except ValueError:
            b_sex = None
    else:
        z_sex_max = z_sex_min = 0.0

    # 로그-바운드
    raw_lb = float(bounds.get("lower", 0.0))
    lb  = None if raw_lb <= 0.0 else np.log(raw_lb)
    ub  = np.log(float(bounds["upper"]))

    # log(mu) 기본 상/하한
    log_vals = at.log(pm.math.clip(mu_age, 1e-9, 1.0))
    log_max  = at.max(log_vals)
    log_min  = at.min(log_vals)

    has_U = U.shape[1] > 0

    # (a) RE 기여: 레벨별 max/min 누적 (보수적)
    if (alpha is not None) and has_U:
        loc_ids = np.asarray(pm_model.coords[re_dim], dtype=int)
        levels  = np.array([G.nodes[i]['level'] for i in loc_ids], dtype=int)
        lvls_of_interest = np.unique(levels[(levels > ref_level) & (levels <= max_level)])

        for lvl in lvls_of_interest:
            idx = np.nonzero(levels == lvl)[0]       # numpy index
            if idx.size:
                sub = at.take(alpha, idx)        # (k_l,)
                log_max = log_max + at.max(sub)
                log_min = log_min + at.min(sub)

    # (b) 성별 고정효과 기여
    if b_sex is not None:
        try:
            log_max = log_max + z_sex_max * b_sex
            log_min = log_min + z_sex_min * b_sex
        except Exception:
            b_sex_f = at.as_tensor_variable(float(b_sex))
            log_max = log_max + z_sex_max * b_sex_f
            log_min = log_min + z_sex_min * b_sex_f

    # (c) 위반량(양수) 계산
    v_low  = at.as_tensor_variable(0.0) if lb is None else at.maximum(0.0, lb  - log_min)
    v_high = at.maximum(0.0, log_max - ub)

    # (d) 강한 Normal(0, 1e-6) 페널티
    sigma   = 1e-6
    v_stack = at.stack([v_low, v_high])
    logp    = at.sum(pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), v_stack))

    return pm.Potential(f"covariate_constraint_{data_type}", logp)
