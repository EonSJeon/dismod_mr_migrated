import numpy as np
import pandas as pd
import pymc as pm
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at
import warnings

SEX_VALUE = {1: .5, 3: 0., 2: -.5}


def MyTruncatedNormal(name, mu, sigma, lower, upper):
    # 1) latent unconstrained
    z = pm.Normal(f"{name}_z", mu=0, sigma=1)
    # 2) map into [lower,upper]
    sigma = pm.Deterministic(name,
        lower + (upper - lower) * pm.math.sigmoid(z)
    )
    # 3) compute logZ and jacobian
    sqrt2 = np.sqrt(2.0)
    a = (lower - mu) / (sigma * sqrt2)
    b = (upper - mu) / (sigma * sqrt2)
    logZ = at.log(0.5 * (at.erf(b) - at.erf(a)))
    logp = (
        -0.5 * ((sigma - mu)/sigma)**2
        - at.log(sigma * at.sqrt(2*np.pi))
        - logZ
        + at.log((upper-lower) * pm.math.sigmoid(z)*(1-pm.math.sigmoid(z)))
    )
    # 4) inject as potential
    pm.Potential(f"{name}_trunc", logp)
    return sigma

def build_random_effects_matrix(
    data_type: str,
) -> tuple[pd.DataFrame, pd.Series]:
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    input_data_dt = sd.get(f'input_data_{data_type}')
    if input_data_dt is None:
        raise KeyError(f"shared_data['input_data_{data_type}'] not set.")
    G: nx.DiGraph = sd['region_id_graph']
    root_id = sd['global_id']
    reference_area_id = sd['reference_area_id']

    n = len(input_data_dt)
    nodes = list(G.nodes)

    # --- 1) Build U (cache shortest paths per unique location) ---
    U = pd.DataFrame(0.0, index=input_data_dt.index, columns=nodes)
    loc_series = input_data_dt['location_id'].astype(int)
    unique_locs = loc_series.unique()
    path_cache: dict[int, list[int]] = {
        loc: nx.shortest_path(G, root_id, loc) if loc in G else None
        for loc in unique_locs
    }

    for idx, loc in loc_series.items():
        path = path_cache.get(loc)
        if not path:
            continue
        # 벡터화 할당
        U.loc[idx, path] = 1.0

    # 빈 경우에도 저장 후 반환
    if U.empty or U.values.sum() == 0:
        U_ref = pd.Series(dtype=float)
        sd[f'U_{data_type}'] = U
        sd[f'U_ref_{data_type}'] = U_ref
        return U, U_ref

    # --- 2) keep only nodes below reference level & with variation (or Constant RE) ---
    base_level = G.nodes[reference_area_id]['level']

    keep_consts: set[int] = set()
    for k, spec in (sd['parameters'][data_type].get('random_effects', {}) or {}).items():
        if isinstance(spec, dict) and spec.get('dist') == 'Constant':
            try:
                keep_consts.add(int(k))
            except Exception:
                # 키가 숫자 id가 아니면 매칭 불가 → 무시
                pass

    cols = [
        c for c in nodes
        if (c in U.columns)
        and (U[c].sum() > 0)
        and (G.nodes[c]['level'] > base_level)
        and (1 <= U[c].sum() < n or c in keep_consts)
    ]
    U = U[cols].copy()

    # --- 3) Centering vector: subtract reference path so ref has net zero effect ---
    path_to_ref = set(nx.shortest_path(G, root_id, reference_area_id))
    shifts = {c: 1.0 if c in path_to_ref else 0.0 for c in U.columns}
    U_ref = pd.Series(shifts, index=U.columns)

    U = U.sub(U_ref, axis=1)

    # --- 4) save & return ---
    sd[f'U_{data_type}'] = U
    sd[f'U_ref_{data_type}'] = U_ref
    return U, U_ref

def build_sigma_alpha(
    data_type: str,
) -> List[Any]:
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data
    parameters = sd['parameters']
    params_dt = parameters[data_type]
    re_specs = params_dt.get('random_effects', {})

    sigma_alpha: List[Any] = []
    max_depth = sd['max_depth']

    for i in range(max_depth):
        name = f'sigma_alpha_{data_type}_{i}'
        spec = re_specs.get(name)

        if spec:
            mu = float(spec['mu'])
            s0 = max(float(spec['sigma']), 1e-3)
            lb = min(mu, spec['lower'])
            ub = max(mu, spec['upper'])
        else:
            mu = 0.05
            s0 = 0.03
            lb = 0.05
            ub = 0.5

        sigma_alpha.append(
            MyTruncatedNormal(
                name=name,
                mu=mu,
                sigma=s0,
                lower=lb,
                upper=ub
            )
        )

    sd[f'sigma_alpha_{data_type}'] = sigma_alpha
    return sigma_alpha

def build_alpha(data_type: str) -> list[Any]:
    """
    Create per-node random effects alpha for U_{data_type}, enforcing sibling
    sum-to-zero without re-registering variables. Returns alpha_list aligned to U.columns.
    """
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    G: nx.DiGraph   = sd['region_id_graph']
    U: pd.DataFrame = sd[f'U_{data_type}']
    sigma_alpha     = sd[f'sigma_alpha_{data_type}']
    params_dt       = sd['parameters'][data_type]
    specs           = params_dt.get('random_effects', {}) or {}
    sum_zero_re     = bool(params_dt.get('sum_zero_re', params_dt.get('zero_re', True)))

    if U.shape[1] == 0:
        sd[f'alpha_{data_type}'] = []
        return []

    cols = list(U.columns)

    def _get_spec(c):
        return specs.get(c, specs.get(str(c), specs.get(int(c), None)))

    def _sigma_for(c):
        return sigma_alpha[G.nodes[c]['level']]

    def _make_alpha(c):
        sp   = _get_spec(c)
        name = f'alpha_{data_type}_{c}'
        if sp:
            dist = sp.get('dist')
            if dist == 'Normal':
                return pm.Normal(name, mu=float(sp.get('mu', 0.0)),
                                 sigma=float(sp.get('sigma', 1.0)), initval=0.0)
            elif dist == 'TruncatedNormal':
                return MyTruncatedNormal(
                    name=name,
                    mu=float(sp['mu']),
                    sigma=max(float(sp['sigma']), 1e-3),
                    lower=float(sp['lower']),
                    upper=float(sp['upper']),
                )
            elif dist == 'Constant':
                return float(sp.get('mu', 0.0))
            else:
                raise ValueError(f"Unknown dist {dist!r} for {name}")
        return pm.Normal(name, mu=0.0, sigma=_sigma_for(c), initval=0.0)

    # 1) pivot plan
    pivot_plan: dict[int, tuple[list[int], float]] = {}
    if sum_zero_re:
        for p in G.nodes:
            sibs = [c for c in G.successors(p) if c in cols]
            if len(sibs) < 2:
                continue
            const_sum = 0.0
            free = []
            for c in sibs:
                sp = _get_spec(c)
                if sp and sp.get('dist') == 'Constant':
                    const_sum += float(sp.get('mu', 0.0))
                else:
                    free.append(c)
            if len(free) >= 2:
                pivot, rest = free[0], free[1:]
                pivot_plan[pivot] = (rest, const_sum)

    pivot_nodes = set(pivot_plan.keys())

    # 2) non-pivot 생성
    alpha_map: dict[int, Any] = {}
    for c in cols:
        if c in pivot_nodes:
            continue
        alpha_map[c] = _make_alpha(c)

    # 3) pivot = -(rest + const_sum)
    for pivot, (rest, const_sum) in pivot_plan.items():
        for r in rest:
            if r not in alpha_map:
                alpha_map[r] = _make_alpha(r)
        sum_rest = sum(alpha_map[r] for r in rest) if rest else 0.0
        alpha_map[pivot] = pm.Deterministic(
            f'alpha_{data_type}_{pivot}',
            -(sum_rest + const_sum)
        )

    alpha_list = [alpha_map[c] for c in cols]
    sd[f'alpha_{data_type}'] = alpha_list
    return alpha_list

def build_fixed_effects_design(data_type: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    input_data_dt: pd.DataFrame = sd[f"input_data_{data_type}"].copy()

    # 1) X 구성
    keep = [c for c in input_data_dt.columns if c.startswith('x_')]
    X = input_data_dt[keep].copy()
    # x_sex 추가
    if 'sex_id' not in input_data_dt.columns:
        raise ValueError("input_data must contain 'sex_id' to build x_sex.")
    X['x_sex'] = [SEX_VALUE[int(row['sex_id'])] for _, row in input_data_dt.iterrows()]
    X = X.astype(float)

    # 2) ESS 가중치 확인
    if 'effective_sample_size' not in input_data_dt.columns:
        raise ValueError("'effective_sample_size' 컬럼이 필요합니다.")
    w = input_data_dt['effective_sample_size'].astype(float)
    if w.isna().any():
        raise ValueError("'effective_sample_size'에 NA 값이 있습니다. 모든 값이 유효해야 합니다.")
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Sum of effective_sample_size must be positive and finite.")

    # 3) 가중 평균(centering)
    X_centering = X.mul(w, axis=0).sum(axis=0) / w_sum

    # 4) 가중 표준편차(바닥값 포함)
    def _wstd(col: pd.Series, w: pd.Series) -> float:
        m = (col * w).sum() / w.sum()
        var = (w * (col - m) ** 2).sum() / w.sum()
        return float(np.sqrt(var))

    X_scaling = X.apply(lambda c: max(_wstd(c, w), 1e-12))

    # 5) 표준화
    X_std = (X - X_centering) / X_scaling

    # 공유 저장
    sd[f'X_{data_type}']           = X_std
    sd[f'X_centering_{data_type}'] = X_centering
    sd[f'X_scaling_{data_type}']   = X_scaling
    return X_std, X_centering, X_scaling

def build_beta(data_type: str, X: pd.DataFrame) -> List[Any]:
    """
    X.columns 순서에 맞춰 고정효과 계수 목록(beta)을 생성한다.
    반환: beta(list) — PyMC RV 또는 Deterministic 또는 float(상수)
    부수효과: pm_model.shared_data['beta'] 저장.
    """
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data
    params_dt = sd['parameters'][data_type]
    fe_specs  = (params_dt.get('fixed_effects') or {})

    beta: List[Any] = []

    for effect in X.columns:
        name = f'beta_{data_type}_{effect}'
        spec = fe_specs.get(effect)

        if spec is None:
            # 기본 Normal(0,1)
            beta.append(pm.Normal(name, mu=0.0, sigma=1.0))
            continue

        dist = spec.get('dist', 'Normal')

        if dist == 'Constant':
            val = float(spec.get('mu', 0.0))
            beta.append(pm.Deterministic(name, at.as_tensor_variable(val)))

        elif dist == 'TruncatedNormal':
            beta.append(
                MyTruncatedNormal(
                    name=name,
                    mu=float(spec['mu']),
                    sigma=max(float(spec['sigma']), 1e-3),
                    lower=float(spec['lower']),
                    upper=float(spec['upper']),
                )
            )

        elif dist == 'HalfNormal':
            sign = spec.get('sign', 'positive')
            half = pm.HalfNormal(
                name + ("_half" if sign == 'negative' else ""),
                sigma=max(float(spec.get('sigma', 1.0)), 1e-3),
                initval=abs(spec.get('initval', 0.1)),
            )
            if sign == 'negative':
                beta.append(pm.Deterministic(name, -half))
            else:
                beta.append(half)

        elif dist == 'Normal':
            beta.append(
                pm.Normal(
                    name,
                    mu=float(spec.get('mu', 0.0)),
                    sigma=float(spec.get('sigma', 1.0)),
                )
            )
        else:
            raise ValueError(f"Unknown fixed_effects dist '{dist}' for effect '{effect}'.")

    # 공유 저장
    sd[f'beta_{data_type}'] = beta
    return beta

def mean_covariate_model(data_type: str, mu: at.TensorVariable):

    # -------- Random effects --------
    U, _ = build_random_effects_matrix(data_type)  # sd[f"U_{data_type}"]도 내부에서 저장된다고 가정
    build_sigma_alpha(data_type)                   # sd[f"sigma_alpha_{data_type}"] 세팅
    alpha = build_alpha(data_type)                 # sd[f"alpha_{data_type}"] 세팅

    # -------- Fixed effects design --------
    X, _, _ = build_fixed_effects_design(data_type)
    beta = build_beta(data_type, X)

    # -------- Linear predictor --------
    n_obs = U.shape[0]  # U는 관측 수 x 노드 수. 노드가 0개여도 행 수는 관측 수로 유지됨.

    if alpha:
        alpha_stack = pm.math.stack(alpha)
        rand_term   = pm.math.dot(U.values, alpha_stack)
    else:
        rand_term   = at.zeros((n_obs,))

    if beta:
        beta_stack  = pm.math.stack(beta)
        fix_term    = pm.math.dot(X.values, beta_stack)
    else:
        fix_term    = at.zeros((n_obs,))

    # -------- pi --------
    pi = mu * pm.math.exp(rand_term + fix_term)
    pm.Deterministic(f"pi_{data_type}", pi)

    return pi


def dispersion_covariate_model(
    data_type: str,
    delta_lb: float,
    delta_ub: float,
):
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data
    input_data_dt = sd[f"input_data_{data_type}"]
    n_obs = len(input_data_dt)

    if not (np.isfinite(delta_lb) and np.isfinite(delta_ub) and delta_lb > 0 and delta_ub > delta_lb):
        raise ValueError(f"Invalid delta bounds: lb={delta_lb}, ub={delta_ub} (require 0 < lb < ub)")

    lower = float(np.log(delta_lb))
    upper = float(np.log(delta_ub))

    eta = pm.Uniform(
        name=f"eta_{data_type}",
        lower=lower,
        upper=upper,
        initval=0.5 * (lower + upper),
    )

    keep_cols = [c for c in input_data_dt.columns if c.startswith("z_")]
    Z = input_data_dt[keep_cols].select_dtypes(include=[np.number]).copy()
    has_Z = Z.shape[1] > 0

    dropped = []
    if has_Z:
        # fill NaN with 0 (no contribution)
        Z = Z.fillna(0.0)
        # drop constant columns (std==0)
        const_mask = (Z.std(ddof=0) == 0.0)
        if const_mask.any():
            dropped = Z.columns[const_mask].tolist()
            Z = Z.loc[:, ~const_mask]

    cov_dim = f"z_covariate_{data_type}"
    obs_dim = f"obs_dim_{data_type}"

    pm_model.add_coord(obs_dim, np.arange(n_obs), mutable=False)
        
    
    if has_Z:
        # register covariate coord (or validate identical)
        cols = Z.columns.tolist()
        pm_model.add_coord(cov_dim, cols, mutable=False)

        # prior on coefficients
        zeta = pm.Normal(
            name=f"zeta_{data_type}",
            mu=0.0,
            sigma=0.25,
            dims=(cov_dim,),
            initval=np.zeros(Z.shape[1]),
        )

        Z_mat = np.asarray(Z.values, dtype=float)
        delta = pm.Deterministic(
            name=f"delta_{data_type}",
            var=pm.math.exp(eta + pm.math.dot(Z_mat, zeta)),
            dims=(obs_dim,),
        )
    else:
        if dropped:
            warnings.warn(f"[dispersion] dropped constant z_* columns: {dropped}", RuntimeWarning)
        # no z-covariates → constant delta per obs
        delta = pm.Deterministic(
            name=f"delta_{data_type}",
            var=pm.math.exp(eta) * at.ones(n_obs),
            dims=(obs_dim,),
        )

    return delta
