import pymc as pm
import numpy as np
import networkx as nx


def find_spline_initial_vals(vars, method, tol, verbose):
    """
    Generate initial values for spline knots sequentially via MAP.

    Parameters
    ----------
    vars : dict
      Model variables (including 'gamma' list)
    method : str
      Optimization method name (e.g., 'BFGS', 'Newton')
    tol : float
      Convergence tolerance passed to find_MAP
    verbose : bool
      If True, print progress
    """
    base_vars = [
        vars.get('p_obs'),
        vars.get('pi_sim'),
        vars.get('parent_similarity'),
        vars.get('mu_sim'),
        vars.get('mu_age_derivative_potential'),
        vars.get('covariate_constraint'),
    ]
    for i, knot in enumerate(vars.get('gamma', [])):
        if verbose:
            print(f"Fitting first {i+1} of {len(vars['gamma'])} spline knots...")
        fit_vars = [v for v in base_vars if v is not None] + [knot]
        # Perform MAP optimization
        map_vals = pm.find_MAP(vars=fit_vars, method=method, tol=tol)
        # Update test_values for next initial guess
        for v in fit_vars:
            name = getattr(v, 'name', None)
            if name and name in map_vals:
                # set Aesara test value
                try:
                    v.tag.test_value = map_vals[name]
                except Exception:
                    pass
        if verbose:
            print_mare(vars)


def find_re_initial_vals(vars, method, tol, verbose):
    """
    Initialize random-effect alpha values and variances via MAP over hierarchy.
    """
    if 'hierarchy' not in vars:
        return
    # column mapping for random effects
    if 'U' not in vars:
        return
    col_map = {col: idx for idx, col in enumerate(vars['U'].columns)}

    # repeated BFS fitting
    for _ in range(3):
        for parent in nx.bfs_tree(vars['hierarchy'], 'all'):
            children = list(vars['hierarchy'].successors(parent))
            if not children:
                continue
            base_vars = [
                vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'),
                vars.get('parent_similarity'), vars.get('mu_sim'),
                vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')
            ]
            base_vars = [v for v in base_vars if v is not None]
            # collect alpha nodes for this parent + children
            re_nodes = []
            for node in children + [parent]:
                if node in col_map:
                    alpha_node = vars['alpha'][col_map[node]]
                    re_nodes.append(alpha_node)
            if not re_nodes:
                continue
            fit_vars = base_vars + re_nodes
            map_vals = pm.find_MAP(vars=fit_vars, method=method, tol=tol)
            for v in fit_vars:
                name = getattr(v, 'name', None)
                if name and name in map_vals:
                    try:
                        v.tag.test_value = map_vals[name]
                    except Exception:
                        pass
    # final fit for sigma_alpha
    base_vars = [
        vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'),
        vars.get('parent_similarity'), vars.get('mu_sim'),
        vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')
    ]
    base_vars = [v for v in base_vars if v is not None]
    sigma_vars = vars.get('sigma_alpha', [])
    fit_vars = base_vars + sigma_vars
    map_vals = pm.find_MAP(vars=fit_vars, method=method, tol=tol)
    for v in fit_vars:
        name = getattr(v, 'name', None)
        if name and name in map_vals:
            try:
                v.tag.test_value = map_vals[name]
            except Exception:
                pass
    if verbose:
        print_mare(vars)


def find_fe_initial_vals(vars, method, tol, verbose):
    """Initialize fixed-effect beta values via MAP."""
    base_vars = [
        vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'),
        vars.get('parent_similarity'), vars.get('mu_sim'),
        vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')
    ]
    base_vars = [v for v in base_vars if v is not None]
    beta_vars = vars.get('beta', [])
    fit_vars = base_vars + beta_vars
    map_vals = pm.find_MAP(vars=fit_vars, method=method, tol=tol)
    for v in fit_vars:
        name = getattr(v, 'name', None)
        if name and name in map_vals:
            try:
                v.tag.test_value = map_vals[name]
            except Exception:
                pass
    if verbose:
        print_mare(vars)


def find_dispersion_initial_vals(vars, method, tol, verbose):
    """Initialize dispersion parameters eta and zeta via MAP."""
    base_vars = [
        vars.get('p_obs'), vars.get('pi_sim'), vars.get('smooth_gamma'),
        vars.get('parent_similarity'), vars.get('mu_sim'),
        vars.get('mu_age_derivative_potential'), vars.get('covariate_constraint')
    ]
    base_vars = [v for v in base_vars if v is not None]
    disp_vars = []
    if 'eta' in vars:
        disp_vars.append(vars['eta'])
    if 'zeta' in vars:
        disp_vars.append(vars['zeta'])
    fit_vars = base_vars + disp_vars
    map_vals = pm.find_MAP(vars=fit_vars, method=method, tol=tol)
    for v in fit_vars:
        name = getattr(v, 'name', None)
        if name and name in map_vals:
            try:
                v.tag.test_value = map_vals[name]
            except Exception:
                pass
    if verbose:
        print_mare(vars)
