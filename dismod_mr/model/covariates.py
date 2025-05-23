import numpy as np
import pandas as pd
import pymc as pm
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at

SEX_VALUE = {'male': .5, 'total': 0., 'female': -.5}

def debug_hierarchy(hierarchy):
    """
    Print detailed information about a NetworkX DiGraph without using nx.info(),
    so it works even if that function isn’t available.
    """
    # 1) 간단한 요약
    n_nodes = hierarchy.number_of_nodes()
    n_edges = hierarchy.number_of_edges()
    print(f"Hierarchy summary: {n_nodes} nodes, {n_edges} edges\n")

    # 2) 노드 리스트
    print("Nodes:")
    for node in hierarchy.nodes():
        print(f"  - {node}")
    print()

    # 3) 엣지 리스트 (parent -> child)
    print("Edges:")
    for parent, child in hierarchy.edges():
        print(f"  - {parent} → {child}")
    print()

    # 4) 인접 리스트
    print("Adjacency list:")
    for node, neighbors in hierarchy.adjacency():
        nbrs = list(neighbors)
        print(f"  - {node}: {nbrs}")
    print()


def build_random_effects_matrix(
    input_data: pd.DataFrame,
    model: Any,
    root_area: str,
    parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct the random-effects indicator matrix U and its shift vector U_shift.

    Parameters
    ----------
    input_data : pd.DataFrame
        Observations with an 'area' column indicating location.
    model : Any
        Object with `hierarchy` attribute (a networkx.DiGraph) describing the area tree.
    root_area : str
        Name of the reference node in the hierarchy.
    parameters : dict
        Model parameters, expects 'random_effects' key mapping names to specs.

    Returns
    -------
    U : pd.DataFrame
        Indicator matrix of shape (n_obs, n_effects) for random effects.
    U_shift : pd.Series
        Shift vector for centering, indexed by the same columns as U.
    """
    n = len(input_data)
    hierarchy = model.hierarchy

    # debug_hierarchy(hierarchy)
    nodes = list(hierarchy.nodes)
    U = pd.DataFrame(0.0, index=input_data.index, columns=nodes)

    for idx, area in input_data['area'].items():
        if area not in hierarchy:
            print(f'WARNING: "{area}" not in model hierarchy, skipping')
            continue
        path = nx.shortest_path(hierarchy, 'all', area)
        for lvl, node in enumerate(path):
            hierarchy.nodes[node]['level'] = lvl
            U.at[idx, node] = 1.0

    for node in nodes:
        path = nx.shortest_path(hierarchy, 'all', node)
        for lvl, nd in enumerate(path):
            hierarchy.nodes[nd]['level'] = lvl

    if U.empty:
        return U, pd.Series(dtype=float)

    base_level = hierarchy.nodes[root_area]['level']
    cols = [
        c for c in nodes
        if U[c].any() and hierarchy.nodes[c]['level'] > base_level
    ]
    U = U[cols]

    keep_consts = [
        name for name, spec in parameters.get('random_effects', {}).items()
        if spec.get('dist') == 'Constant'
    ]

    valid = [c for c in U.columns if 1 <= U[c].sum() < n or c in keep_consts]
    U = U[valid].copy()

    path_to_root = nx.shortest_path(hierarchy, 'all', root_area)
    U_shift = pd.Series(
        {c: (1.0 if c in path_to_root else 0.0)
         for c in U.columns},
        dtype=float)

    U = U.sub(U_shift, axis=1)
    return U, U_shift


def build_sigma_alpha(data_type: str,
                      parameters: Dict[str, Any],
                      max_depth: int = 5) -> List[Any]:
    """
    Generate TruncatedNormal priors for hierarchical sigma_alpha.
    """
    sigma_alpha: List[Any] = []
    re_specs = parameters.get('random_effects', {})
    for i in range(max_depth):
        name = f'sigma_alpha_{data_type}_{i}'
        spec = re_specs.get(name)
        if spec:
            mu = float(spec['mu'])
            tau = max(float(spec.get('sigma', 1e-3)), 1e-3)**-2
            lb = min(mu, spec['lower'])
            ub = max(mu, spec['upper'])
        else:
            mu, tau, lb, ub = 0.05, 0.03**-2, 0.05, 0.5
        sigma = 1.0 / np.sqrt(tau)
        sigma_alpha.append(
            pm.TruncatedNormal(name=name,
                               mu=mu,
                               sigma=sigma,
                               lower=lb,
                               upper=ub,
                               initval=mu,
                               transform=None))
    return sigma_alpha


def build_alpha(
    data_type: str,
    U: pd.DataFrame,
    sigma_alpha: List[Any],
    parameters: Dict[str, Any],
    zero_re: bool,
    hierarchy: nx.DiGraph,
) -> Tuple[List[Any], List[float], List[Any]]:
    """
    Generate random-effect coefficients alpha, constants and potentials.
    """
    alpha: List[Any] = []
    const_alpha_sigma: List[float] = []
    alpha_potentials: List[Any] = []
    if U.shape[1] == 0:
        return alpha, const_alpha_sigma, alpha_potentials

    tau_list = [
        1.0 / (sigma_alpha[hierarchy.nodes[c]['level']].distribution.sigma**2)
        for c in U.columns
    ]
    for col, tau in zip(U.columns, tau_list):
        name = f'alpha_{data_type}_{col}'
        spec = parameters.get('random_effects', {}).get(col)
        if spec:
            dist = spec['dist']
            if dist == 'Normal':
                mu0, s0 = float(spec['mu']), float(spec['sigma'])
                rv = pm.Normal(name, mu=mu0, sigma=s0, initval=0.0)
            elif dist == 'TruncatedNormal':
                mu0 = float(spec['mu'])
                tau0 = max(float(spec['sigma']), 1e-3)**-2
                rv = pm.TruncatedNormal(name,
                                        mu=mu0,
                                        sigma=1 / np.sqrt(tau0),
                                        lower=spec['lower'],
                                        upper=spec['upper'],
                                        initval=0.0)
            elif dist == 'Constant':
                rv = float(spec['mu'])
            else:
                raise ValueError(f"Unknown dist {dist} for {name}")
        else:
            rv = pm.Normal(name, mu=0.0, sigma=1.0 / np.sqrt(tau), initval=0.0)
        alpha.append(rv)
        const_alpha_sigma.append(
            float(spec.get('sigma', np.nan)
                  ) if spec and spec.get('dist') == 'Constant' else np.nan)

    if zero_re:
        idx_map = {c: i for i, c in enumerate(U.columns)}
        for parent in hierarchy.nodes:
            children = [
                c for c in hierarchy.successors(parent) if c in idx_map
            ]
            if not children: continue
            i0 = idx_map[children[0]]
            spec0 = parameters.get('random_effects', {}).get(children[0])
            if spec0 and spec0.get('dist') == 'Constant': continue
            sibs = [idx_map[c] for c in children[1:]]
            det = pm.Deterministic(f'alpha_det_{data_type}_{i0}',
                                   -sum(alpha[i] for i in sibs))
            old = alpha[i0]
            alpha[i0] = det
            if isinstance(old, pm.Distribution):
                alpha_potentials.append(
                    pm.Potential(f'alpha_pot_{data_type}_{children[0]}',
                                 old.logp(det)))
    return alpha, const_alpha_sigma, alpha_potentials


def mean_covariate_model(data_type: str,
                         mu,
                         input_data: pd.DataFrame,
                         parameters: dict,
                         model,
                         root_area: str,
                         root_sex: str,
                         root_year,
                         zero_re: bool = True) -> dict:
    """
    공변량(고정효과)과 랜덤효과를 포함한 예측 변수(pi)를 생성하는 함수

    매개변수:
    - data_type: 변수명 접두사로 사용할 문자열
    - mu: 기준 평균값 (스칼라 혹은 모집단 수준 효과)
    - input_data: 관측 데이터를 담은 DataFrame
    - parameters: 사용자 지정 priors 설정 사전
    - model: ModelData 객체 (hierarchy, output_template 포함)
    - root_area, root_sex, root_year: 기준(reference) 영역, 성별, 연도
    - zero_re: True면 참조집단 랜덤효과를 0으로 고정

    반환값:
    고정효과 및 랜덤효과, 예측값 pi 등을 담은 dict
    """
    # --- 1) 랜덤 효과 행렬 생성 및 shift 벡터 계산 ---
    U, U_shift = build_random_effects_matrix(input_data, model, root_area, parameters)
    sigma_alpha = build_sigma_alpha(data_type, parameters)
    alpha, const_alpha_sigma, alpha_potentials = build_alpha(
        data_type, U, sigma_alpha, parameters, zero_re, model.hierarchy)


    # --- 2) 고정효과(covariate) 행렬 구성 ---
    keep = [c for c in input_data.columns if c.startswith('x_')]
    X = input_data[keep].copy()
    X['x_sex'] = [SEX_VALUE[row['sex']] for _, row in input_data.iterrows()]

    # --- 3) 센터링을 위한 shift 계산 ---
    X_shift = pd.Series(0.0, index=X.columns)
    # output_template에서 area, sex, year별 평균 추출
    tpl = model.output_template.groupby(['area', 'sex', 'year']).mean()
    # covariate 및 pop 열만 선택 (없는 열은 reindex로 0으로 채움)
    covs = tpl.reindex(columns=list(X.columns) + ['pop'], fill_value=0)


    # 계층구조에서 리프 노드(하위 노드 없는 영역) 추출
    leaves = [n for n in nx.bfs_tree(model.hierarchy, root_area)
              if model.hierarchy.out_degree(n) == 0] or [root_area]
    # 참조 조건에 따라 leaf_cov 선택
    if root_sex == 'total' and root_year == 'all':
        cov_tmp = covs.reset_index().drop(['sex', 'year'], axis=1)
        leaf_cov = cov_tmp.groupby('area').mean().loc[leaves]
    else:
        leaf_cov = covs.loc[[(l, root_sex, root_year) for l in leaves]]

    # template에 존재하는 covariate만 shift 반영, 없으면 0 유지
    for cov in X.columns:
        if cov in leaf_cov.columns:
            X_shift[cov] = (leaf_cov[cov] * leaf_cov['pop']).sum() / leaf_cov['pop'].sum()
        else:
            X_shift[cov] = 0.0
    # 센터링 적용
    X = X - X_shift

    # --- 4) 고정효과 priors 설정 ---
    beta = []
    const_beta_sigma = []
    for effect in X.columns:
        name = f'beta_{data_type}_{effect}'
        spec = parameters.get('fixed_effects', {}).get(effect)
        if spec:
            dist = spec['dist']
            if dist == 'TruncatedNormal':
                tau = spec['sigma']**-2
                beta.append(
                    pm.TruncatedNormal(name,
                                       mu=spec['mu'],
                                       sigma=1/np.sqrt(tau),
                                       lower=spec['lower'],
                                       upper=spec['upper']))
            else:
                # 기본 Normal 분포 사용
                beta.append(pm.Normal(name,
                                       mu=spec.get('mu', 0),
                                       sigma=spec.get('sigma', 1)))
            # Constant인 경우 sigma 기록, 아니면 NaN
            const_beta_sigma.append(spec.get('sigma') if dist=='Constant' else np.nan)
        else:
            # 사전 설정 없으면 표준 Normal
            beta.append(pm.Normal(name, mu=0.0, sigma=1.0))
            const_beta_sigma.append(np.nan)


    n_obs = U.shape[0]

    # 1) stack only if non-empty, else make a zeros vector
    if alpha:
        alpha_stack = pm.math.stack(alpha)         # shape (n_re,)
        rand_term   = pm.math.dot(U.values, alpha_stack)
    else:
        rand_term   = at.zeros((n_obs,))

    if beta:
        beta_stack = pm.math.stack(beta)           # shape (n_fx,)
        fix_term   = pm.math.dot(X.values, beta_stack)
    else:
        fix_term   = at.zeros((n_obs,))

    # 2) combine them just like NumPy would
    pi = pm.Deterministic(
        f"pi_{data_type}",
        mu * pm.math.exp(rand_term + fix_term)
    )

    # 결과 dict 반환
    return {
        'pi': pi,
        'U': U,
        'U_shift': U_shift,
        'sigma_alpha': sigma_alpha,
        'alpha': alpha,
        'alpha_potentials': alpha_potentials,
        'const_alpha_sigma': const_alpha_sigma,
        'X': X,
        'X_shift': X_shift,
        'beta': beta,
        'const_beta_sigma': const_beta_sigma,
        'hierarchy': model.hierarchy
    }



def dispersion_covariate_model(name: str, input_data: pd.DataFrame,
                               delta_lb: float,
                               delta_ub: float) -> Dict[str, Any]:
    """
    Generate dispersion (delta) covariate model.

    Returns dict with:
      - eta : Uniform prior on log(delta)
      - Z : DataFrame of covariates
      - zeta (optional) : Normal coeffs for Z
      - delta : Deterministic exp(eta + Z @ zeta)
    """
    lower, upper = np.log(delta_lb), np.log(delta_ub)
    eta = pm.Uniform(f'eta_{name}',
                     lower=lower,
                     upper=upper,
                     initval=0.5 * (lower + upper))

    # select non-constant Z columns
    cols = [
        c for c in input_data.columns
        if c.startswith('z_') and input_data[c].std() > 0
    ]
    Z = input_data[cols].copy()
    if len(Z.columns) > 0:
        zeta = pm.Normal(f'zeta_{name}',
                         mu=0.0,
                         sigma=0.25,
                         dims=["covariate"],
                         initval=np.zeros(len(Z.columns)))
        delta = pm.Deterministic(
            f'delta_{name}', pm.math.exp(eta + pm.math.dot(Z.values, zeta)))
        return {'eta': eta, 'Z': Z, 'zeta': zeta, 'delta': delta}
    else:
        delta = pm.Deterministic(f'delta_{name}',
                                 pm.math.exp(eta) * np.ones(len(input_data)))
        return {'eta': eta, 'delta': delta}


def predict_for(
    model,
    parameters,
    root_area,
    root_sex,
    root_year,
    area,
    sex,
    year,
    population_weighted: bool,
    vars,
    lower,
    upper
):
    """
    Generate posterior predictive draws for a specific (area, sex, year).

    Returns an array of draws from the model’s posterior predictive distribution,
    optionally aggregating across sub‑areas with population weighting.
    """
    # Number of posterior samples
    mu_node = vars['mu_age']
    mu_trace = mu_node.trace()  # shape: (n_samples, n_ages)
    n_samples = mu_trace.shape[0]

    # Assemble random effect traces
    alpha_trace = np.empty((n_samples, 0))
    if 'alpha' in vars and isinstance(vars['alpha'], list) and vars['alpha']:
        traces = []
        for node, sigma in zip(vars['alpha'], vars['const_alpha_sigma']):
            if hasattr(node, 'trace'):
                traces.append(node.trace())
            else:
                sig = max(sigma, 1e-9)
                traces.append(
                    np.random.normal(loc=float(node), scale=1/np.sqrt(sig), size=n_samples)
                )
        alpha_trace = np.vstack(traces).T

    # Assemble fixed effect traces
    beta_trace = np.empty((n_samples, 0))
    if 'beta' in vars and isinstance(vars['beta'], list) and vars['beta']:
        traces = []
        for node, sigma in zip(vars['beta'], vars['const_beta_sigma']):
            if hasattr(node, 'trace'):
                traces.append(node.trace())
            else:
                sig = max(sigma, 1e-9)
                traces.append(
                    np.random.normal(loc=float(node), scale=1/np.sqrt(sig), size=n_samples)
                )
        beta_trace = np.vstack(traces).T

    # Determine leaves under area
    leaves = [n for n in nx.bfs_tree(model.hierarchy, area)
              if model.hierarchy.out_degree(n) == 0]
    if not leaves:
        leaves = [area]

    # Prepare design matrices
    output = model.output_template.copy()
    grp = output.groupby(['area','sex','year']).mean()

    # Covariates
    if 'X' in vars and not vars['X'].empty:
        X_df = grp.filter(vars['X'].columns).copy()
        if 'x_sex' in X_df.columns:
            X_df['x_sex'] = SEX_VALUE[sex]
        # center
        X_df = X_df - vars['X_shift']
    else:
        X_df = pd.DataFrame(index=grp.index)

    # Random-effects indicator
    if 'U' in vars and not vars['U'].empty:
        U_cols = vars['U'].columns
        U_row = pd.Series(0.0, index=U_cols)
    else:
        U_row = pd.Series(dtype=float)

    # Aggregate
    cov_shift = np.zeros(n_samples)
    total_weight = 0.0
    for leaf in leaves:
        # build U indicator along path
        U_row[:] = 0.0
        path = nx.shortest_path(model.hierarchy, root_area, leaf)
        for node in path[1:]:
            if node in U_row.index:
                U_row[node] = 1.0 - vars['U_shift'].get(node, 0.0)

        log_shift = alpha_trace.dot(U_row.values)

        # fixed effects
        if not beta_trace.size and (leaf, sex, year) in X_df.index:
            x_vals = X_df.loc[(leaf, sex, year)].values
            log_shift += beta_trace.dot(x_vals)

        pop = grp.at[(leaf, sex, year), 'pop']
        if population_weighted:
            cov_shift += np.exp(log_shift) * pop
            total_weight += pop
        else:
            cov_shift += log_shift
            total_weight += 1

    if population_weighted:
        cov_shift /= total_weight
    else:
        cov_shift = np.exp(cov_shift / total_weight)

    # Combine with baseline mu_age draws and clip
    # mu_trace: (n_samples, n_ages)
    # need age index for the target year/sex? here just multiply each draw
    preds = mu_trace * cov_shift[:, None]
    return np.clip(preds, lower, upper)