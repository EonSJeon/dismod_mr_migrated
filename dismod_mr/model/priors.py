import numpy as np
import pymc as pm

import dismod_mr


def similar(
    data_type: str,
    mu_child,
    mu_parent,
    sigma_parent,
    sigma_difference: float,
    offset: float = 1e-9
) -> dict:
    """
    Encode a similarity prior: child mu should be close to parent mu on log scale.

    Returns a potential that penalizes deviation of log(mu_child) from log(mu_parent).
    """
    # determine precision tau
    if hasattr(mu_parent, 'distribution'):
        # mu_parent is a PyMC RV
        tau = 1.0 / (sigma_parent**2 + sigma_difference**2)
    else:
        # mu_parent is numeric array
        denom = ((sigma_parent + offset) / (mu_parent + offset))**2 + sigma_difference**2
        tau = 1.0 / denom

    # log-transform with clipping
    log_child = pm.math.log(pm.math.clip(mu_child, offset, np.inf))
    log_parent = pm.math.log(pm.math.clip(mu_parent, offset, np.inf))

    # use Normal.logp for pointwise log-probabilities
    dist = pm.Normal.dist(mu=log_parent, sigma=1/pm.math.sqrt(tau))
    logp = dist.logp(log_child)

    # sum over observations and add potential
    parent_similarity = pm.Potential(
        name=f'parent_similarity_{data_type}',
        logp=pm.math.sum(logp)
    )

    return {'parent_similarity': parent_similarity}


def level_constraints(
    data_type: str,
    parameters: dict,
    unconstrained_mu_age,
    ages: np.ndarray
) -> dict:
    if 'level_value' not in parameters or 'level_bounds' not in parameters:
        return {}

    lv = parameters['level_value']
    lb = parameters['level_bounds']['lower']
    ub = parameters['level_bounds']['upper']

    # zero-based clip indices
    before_idx = int(np.clip(lv['age_before'] - ages[0], 0, len(ages)))
    after_idx  = int(np.clip(lv['age_after']  - ages[0], 0, len(ages)))
    idx = np.arange(len(ages))

    # build the piecewise‐constant vector, then clip
    base = unconstrained_mu_age
    val  = lv['value']
    seg1 = pm.math.switch(idx < before_idx, val, base)
    seg2 = pm.math.switch(idx > after_idx,  val, seg1)
    clipped = pm.math.clip(seg2, lb, ub)

    # register it as a Deterministic
    mu_age = pm.Deterministic(f'value_constrained_mu_age_{data_type}', clipped)

    # add a similarity‐prior “potential” to keep it near unconstrained
    mu_sim = similar(
        data_type,
        mu_child=mu_age,
        mu_parent=unconstrained_mu_age,
        sigma_parent=0.0,
        sigma_difference=0.01,
        offset=1e-6
    )

    return {
        'mu_age': mu_age,
        'unconstrained_mu_age': unconstrained_mu_age,
        'mu_sim': mu_sim
    }


def covariate_level_constraints(
    name: str,
    model: any,
    vars: dict[str, any],
    ages: np.ndarray
) -> dict:
    """
    Implement priors on covariate-adjusted rate function to enforce level bounds.

    Returns a potential constraint summarizing violations.
    """
    params = model.parameters.get(name, {})
    lvl = params.get('level_value')
    bounds = params.get('level_bounds')
    if not lvl or not bounds:
        return {}

    # sex covariate range after centering
    X_shift = vars['X_shift']
    beta = vars['beta']
    sex_idx = list(X_shift.index).index('x_sex')
    X_sex_max = 0.5 - X_shift['x_sex']
    X_sex_min = -0.5 - X_shift['x_sex']

    # collect U masks by level
    U = vars['U']
    hierarchy = model.hierarchy
    layers = []
    nodes = ['all']
    for _ in range(3):
        nodes = [c for n in nodes for c in hierarchy.successors(n)]
        mask = np.array([col in nodes for col in U.columns])
        if mask.any():
            layers.append(mask)

    # bounds on log(mu_age)
    low = np.log(bounds['lower'])
    high = np.log(bounds['upper'])

    @pm.Potential(name=f'covariate_constraint_{name}')
    def covariate_constraint():
        mu = vars['mu_age']
        log_vals = pm.math.log(mu)
        # base extrema
        log_max = pm.math.max(log_vals)
        log_min = pm.math.min(log_vals)
        # add random effect contributions
        alphas = pm.math.stack(vars['alpha'])
        if alphas.shape[0] > 0:
            for m in layers:
                m_idx = np.where(m)[0]
                log_max += pm.math.max(pm.math.switch(m_idx[:, None]==m_idx, alphas, -np.inf))
                log_min += pm.math.min(pm.math.switch(m_idx[:, None]==m_idx, alphas, np.inf))
        # add sex fixed effect
        b_sex = beta[sex_idx] if isinstance(beta[sex_idx], pm.distributions.Distribution) else beta[sex_idx]
        log_max += X_sex_max * float(b_sex)
        log_min += X_sex_min * float(b_sex)
        # compute violations
        v_low = pm.math.minimum(0, log_min - low)
        v_high = pm.math.maximum(0, log_max - high)
        # penalize quadratically via narrow Normal potential
        sigma = 1e-6
        pot = pm.math.sum(pm.Normal.dist(mu=0, sigma=sigma).logp(pm.math.stack([v_low, v_high])))
        return pot

    return {'covariate_constraint': covariate_constraint}


def derivative_constraints(
    name: str,
    parameters: dict,
    mu_age,
    ages: np.ndarray
) -> dict:
    """
    Implement priors on the derivative of mu_age over specified age intervals.

    parameters must include 'increasing' and 'decreasing' with 'age_start' and 'age_end'.
    Returns a potential penalizing violations of monotonicity.
    """
    inc = parameters.get('increasing')
    dec = parameters.get('decreasing')
    if not inc or not dec:
        return {}

    inc_start = int(np.clip(inc['age_start'] - ages[0], 0, len(ages)-1))
    inc_end = int(np.clip(inc['age_end'] - ages[0], 0, len(ages)-1))
    dec_start = int(np.clip(dec['age_start'] - ages[0], 0, len(ages)-1))
    dec_end = int(np.clip(dec['age_end'] - ages[0], 0, len(ages)-1))

    @pm.Potential(name=f'mu_age_derivative_potential_{name}')
    def mu_age_derivative_potential():
        mu_vals = mu_age
        mu_prime = pm.math.diff(mu_vals)
        inc_violation = pm.math.sum(pm.math.clip(mu_prime[inc_start:inc_end], -np.inf, 0.0))
        dec_violation = pm.math.sum(pm.math.clip(mu_prime[dec_start:dec_end], 0.0, np.inf))
        # strong penalty
        penalty = inc_violation**2 + dec_violation**2
        return -1e12 * penalty

    return {'mu_age_derivative_potential': mu_age_derivative_potential}
