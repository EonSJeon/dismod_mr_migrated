import numpy as np
import pymc as pm
import pytensor.tensor as at
import dismod_mr



def similar(
    data_type: str,
    mu_child: at.TensorVariable,
    mu_parent,
    sigma_parent: float,
    sigma_difference: float,
    offset: float = 1e-9
) -> dict:
    """
    Encode a similarity prior: child mu should be close to parent mu on log scale.
    PyMC 5.3 기준으로 dist.logp() 대신 pm.logp(dist, x) 사용.
    """
    assert pm.modelcontext(None) is not None, 'similar must be called within a PyMC model'
    # 1) tau 계산
    if hasattr(mu_parent, "distribution"):
        # mu_parent가 PyMC RV일 때
        tau = 1.0 / (sigma_parent**2 + sigma_difference**2)
    else:
        # mu_parent가 numeric array일 때
        denom = ((sigma_parent + offset) / (mu_parent + offset))**2 + sigma_difference**2
        tau = 1.0 / denom

    # 2) log-transform with clipping
    log_child  = pm.math.log(pm.math.clip(mu_child, offset, np.inf))
    log_parent = pm.math.log(pm.math.clip(mu_parent, offset, np.inf))

    # 3) Pointwise log‐probabilities: pm.logp(dist, log_child)
    dist = pm.Normal.dist(mu=log_parent, sigma=1 / pm.math.sqrt(tau))
    logp = pm.logp(dist, log_child)  # ← 여기서 dist.logp(log_child) 대신

    # 4) Potential 등록
    parent_similarity = pm.Potential(
        name=f"parent_similarity_{data_type}",
        var=pm.math.sum(logp)
    )

    return {"parent_similarity": parent_similarity}


def level_constraints(
    data_type: str,
    parameters: dict,
    unconstrained_mu_age: at.TensorVariable,
    ages: np.ndarray
) -> dict:
    """
    PyMC 5.3 양식에 맞춘 level_constraints 구현 예시입니다.
    - unconstrained_mu_age: 이미 모델 컨텍스트 안에서 생성된 TensorVariable이어야 합니다.
    - ages: NumPy 배열 ([0,1,2,...]).
    """
    assert pm.modelcontext(None) is not None, 'level_constraints must be called within a PyMC model'
    if "level_value" not in parameters or "level_bounds" not in parameters:
        return {}

    lv = parameters["level_value"]
    lb = parameters["level_bounds"]["lower"]
    ub = parameters["level_bounds"]["upper"]

    # 1) NumPy 단계에서 정수 인덱스 계산
    before_idx = int(np.clip(lv["age_before"] - ages[0], 0, len(ages)))
    after_idx  = int(np.clip(lv["age_after"]  - ages[0], 0, len(ages)))
    i = at.arange(len(ages))  # PyTensor TensorVariable

    # 2) val, lb, ub 를 float으로 변환
    val_f = float(lv["value"])
    lb_f  = float(lb)
    ub_f  = float(ub)

    # 3) piecewise‐constant + clipping (PyTensor 연산)
    with pm.modelcontext(None):  # 이미 상위 코드가 `with pm.Model():` 내부라면, pm.Potential 등이 모델에 등록됩니다
        seg1 = at.switch(i < before_idx, val_f, unconstrained_mu_age)
        seg2 = at.switch(i >  after_idx,  val_f, seg1)
        clipped = at.clip(seg2, lb_f, ub_f)

        # 4) TensorVariable → Deterministic
        mu_age = pm.Deterministic(f"value_constrained_mu_age_{data_type}", clipped)

        # 5) similarity prior 걸기 (위의 similar 함수를 이용)
        sim_dict = similar(
            data_type      = data_type,
            mu_child       = mu_age,
            mu_parent      = unconstrained_mu_age,
            sigma_parent   = 0.0,
            sigma_difference = 0.01,
            offset         = 1e-6
        )
        parent_similarity = sim_dict["parent_similarity"]

    return {
        "mu_age": mu_age,
        "unconstrained_mu_age": unconstrained_mu_age,
        "mu_sim": {"parent_similarity": parent_similarity}
    }



def covariate_level_constraints(
    data_type: str,
    model: any,
    vars: dict[str, any],
    ages: np.ndarray
) -> dict:
    """
    Enforce level‐bounds on the covariate‐adjusted rate curve.
    If bounds['lower'] == 0, we skip the lower‐bound term entirely.
    """

    # 1) Must be inside a `with pm.Model():` context
    if pm.modelcontext(None) is None:
        raise AssertionError('covariate_level_constraints must be called within a PyMC model')

    params = model.parameters.get(data_type, {})
    lvl    = params.get('level_value')
    bounds = params.get('level_bounds')
    if not lvl or not bounds:
        # nothing to do if no level_value or no level_bounds
        return {}

    # 2) Compute sex covariate range (after centering)
    X_shift = vars['X_shift']
    beta    = vars['beta']
    sex_idx = list(X_shift.index).index('x_sex')
    X_sex_max = 0.5 - X_shift['x_sex']
    X_sex_min = -0.5 - X_shift['x_sex']

    # 3) Build “layers” of U‐masks for the random‐effects hierarchy
    U         = vars['U']  # pandas.DataFrame
    hierarchy = model.hierarchy
    layers: list[np.ndarray] = []
    nodes = ['all']
    for _ in range(3):
        nodes = [c for n in nodes for c in hierarchy.successors(n)]
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
    mu = vars['mu_age']  # (TensorVariable, shape=(len(ages),))

    # (a) log(mu) at each age
    log_vals = at.log(mu)

    # (b) base‐curve extrema
    log_max = at.max(log_vals)
    log_min = at.min(log_vals)

    # (c) add random‐effect contributions, if any
    alpha_list = vars.get('alpha', [])
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
    pot = pm.Potential(f"covariate_constraint_{data_type}", var=logp_sum)
    return {"covariate_constraint": pot}










def derivative_constraints(
    data_type: str,
    parameters: dict,
    mu_age: at.TensorVariable,
    ages: np.ndarray
) -> dict:
    """
    Implement priors on the derivative of mu_age over specified age intervals,
    using PyMC 5.3 양식에 맞춰 수정한 버전입니다.

    parameters 에 'increasing'과 'decreasing'이 반드시 있어야 하며,
    해당 구간에서 기울기 위반(penalty)을 계산해 pm.Potential 으로 추가합니다.
    """
    assert pm.modelcontext(None) is not None, 'derivative_constraints must be called within a PyMC model'
    inc = parameters.get("increasing")
    dec = parameters.get("decreasing")
    if not inc or not dec:
        return {}

    # 0) NumPy 단계에서 “증가해야 할 구간”과 “감소해야 할 구간”의 인덱스 계산
    inc_start = int(np.clip(inc["age_start"] - ages[0], 0, len(ages) - 1))
    inc_end   = int(np.clip(inc["age_end"]   - ages[0], 0, len(ages) - 1))
    dec_start = int(np.clip(dec["age_start"] - ages[0], 0, len(ages) - 1))
    dec_end   = int(np.clip(dec["age_end"]   - ages[0], 0, len(ages) - 1))

    # 1) PyMC 모델 컨텍스트(이미 상위 코드에서 `with pm.Model():` 내부여야 함) 하에 연산
    #
    #    PyMC 5.3부터는 `@pm.Potential(...)` 데코레이터 방식이 아니라
    #    “명시적으로 pm.Potential(name=..., var=...)” 형태로 써야 합니다.
    #
    #    여기에서는 “mu_age”가 전에 생성되어 있는 TensorVariable이라고 가정합니다.
    #
    
    # 1-1) 연속 나이별 mu_age 간의 차분 (pytensor.at.diff)
    mu_prime = at.diff(mu_age)  # shape = (len(ages)-1,)

    # 1-2) “증가(increasing)” 구간에서 음수 기울기 위반합
    inc_slice = mu_prime[inc_start:inc_end]  # 해당 슬라이스 구간
    inc_violation = at.sum(at.clip(inc_slice, -np.inf, 0.0))

    # 1-3) “감소(decreasing)” 구간에서 양수 기울기 위반합
    dec_slice = mu_prime[dec_start:dec_end]
    dec_violation = at.sum(at.clip(dec_slice, 0.0, np.inf))

    # 1-4) 벌칙(penalty) 계산 (강하게 벌칙 주기 위해 제곱 후 *1e12)
    penalty = inc_violation**2 + dec_violation**2
    logp = -1e12 * penalty

    # 1-5) pm.Potential 을 명시적으로 생성
    pot = pm.Potential(
        name=f"mu_age_derivative_potential_{data_type}",
        var=logp
    )

    return {"mu_age_derivative_potential": pot}
