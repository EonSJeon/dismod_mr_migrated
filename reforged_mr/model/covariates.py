import numpy as np
import pandas as pd
import pymc as pm
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at
import warnings

SEX_NAME2ID = {'Male': 1, 'Female': 2, 'Both': 3}
SEX_ID2NAME = {v: k for k, v in SEX_NAME2ID.items()}
SEX_NAME2VAL = {'Male': 0.5, 'Female': -0.5, 'Both': 0.0}

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
    # NOTE: Omitting the reference area from the matrix is intentional.
    #       The effect at the reference area itself is for h(a).
    #       Should not be dealt twice.

    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    input_data_dt = sd.get(f'input_data_{data_type}')
    if input_data_dt is None:
        raise KeyError(f"shared_data['input_data_{data_type}'] not set.")

    G: nx.DiGraph = sd['region_id_graph']
    global_id = sd['global_id']
    reference_area_id = sd['reference_area_id']

    n = len(input_data_dt)
    nodes = list(G.nodes)

    # --- 1) Build U (cache shortest paths per unique location) ---
    U = pd.DataFrame(0.0, index=input_data_dt.index, columns=nodes, dtype=float)
    loc_series = input_data_dt['location_id'].astype(int)
    unique_locs = loc_series.unique()
    path_cache: dict[int, list[int]] = {
        loc: nx.shortest_path(G, global_id, loc) if loc in G else None
        for loc in unique_locs
    }

    for idx, loc in loc_series.items():
        path = path_cache.get(loc)
        if not path:
            continue
        U.loc[idx, path] = 1.0

    # 빈 경우에도 저장 후 반환
    if U.empty or U.values.sum() == 0:
        U_ref = pd.Series(dtype=float)
        sd[f'U_{data_type}'] = U
        sd[f'U_ref_{data_type}'] = U_ref
        # coord는 생성하지 않음(빈 축 등록은 피함)
        return U, U_ref

    # --- 2) keep only nodes below reference level & with variation (or Constant RE) ---
    ref_level = G.nodes[reference_area_id]['level']

    keep_consts: set[int] = set()
    for k, spec in (sd['parameters'][data_type].get('random_effects', {}) or {}).items():
        if isinstance(spec, dict) and spec.get('dist') == 'Constant':
            try:
                keep_consts.add(int(k))
            except Exception:
                pass  # 키가 숫자 id가 아니면 무시

    cols = [
        c for c in nodes
        if (c in U.columns)
        and (U[c].sum() > 0)
        and (G.nodes[c]['level'] > ref_level)
        and (1 <= U[c].sum() < n or c in keep_consts)
    ]
    U = U[cols].copy()

    # --- 3) Centering vector: subtract reference path so ref has net zero effect ---
    path_to_ref = set(nx.shortest_path(G, global_id, reference_area_id))
    shifts = {c: 1.0 if c in path_to_ref else 0.0 for c in U.columns}
    U_ref = pd.Series(shifts, index=U.columns)

    U = U.sub(U_ref, axis=1)

    # --- 4) RE 노드 coord 등록 (U.columns 순서가 alpha 벡터 축이 됨) ---
    if U.shape[1] > 0:
        id_dim = f"re_loc_id_{data_type}"
        if id_dim not in pm_model.coords:
            pm_model.add_coord(id_dim, list(U.columns), mutable=False)

    # --- 5) save & return ---
    sd[f'U_{data_type}'] = U
    sd[f'U_ref_{data_type}'] = U_ref

    return U, U_ref

def build_sigma_alpha(data_type: str) -> List[Any]:
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    G: nx.DiGraph = sd['region_id_graph']
    params_dt     = sd['parameters'][data_type]
    re_specs      = params_dt.get('random_effects', {}) or {}

    ref_level = int(G.nodes[sd['reference_area_id']]['level'])
    max_level  = int(sd['max_depth']) - 1

    sigma_alpha: dict[int, Any] = {}
    for lvl in range(ref_level, max_level + 1):
        name    = f'sigma_alpha_{data_type}_{lvl}'
        spec    = re_specs.get(name)

        if spec:
            mu = float(spec['mu'])
            s0 = max(float(spec['sigma']), 1e-3)
            lb = min(mu, float(spec['lower']))
            ub = max(mu, float(spec['upper']))
        else:
            mu, s0, lb, ub = 0.05, 0.03, 0.05, 0.5

        sigma_alpha[lvl] = MyTruncatedNormal(name=name, mu=mu, sigma=s0, lower=lb, upper=ub)

    sd[f'sigma_alpha_{data_type}'] = sigma_alpha
    return sigma_alpha

def build_alpha(data_type: str):
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    G: nx.DiGraph   = sd['region_id_graph']
    U: pd.DataFrame = sd[f'U_{data_type}']
    sigma_alpha     = sd[f'sigma_alpha_{data_type}']  # dict[level] -> RV
    params_dt       = sd['parameters'][data_type]
    specs           = params_dt.get('random_effects', {}) or {}
    sum_zero_re     = bool(params_dt.get('sum_zero_re', params_dt.get('zero_re', True)))

    has_U = U.shape[1] > 0
    if not has_U:
        sd[f'alpha_{data_type}'] = None
        return None

    # --- coord 읽기/등록 (U.columns 순서로 초기화하되, 이미 있으면 그 순서를 사용) ---
    id_dim  = f"re_loc_id_{data_type}"
    loc_ids = list(map(int, pm_model.coords[id_dim]))

    def _get_spec(c):
        return specs.get(c, specs.get(str(c), specs.get(int(c), None)))

    def _sigma_for(c):
        lvl = G.nodes[int(c)]['level']
        try:
            return sigma_alpha[lvl]
        except KeyError:
            raise KeyError(f"sigma_alpha for absolute level {lvl} not found.")

    def _make_alpha(c):
        c = int(c)
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

    # 1) pivot plan (형제 sum-to-zero)
    pivot_plan: dict[int, tuple[list[int], float]] = {}
    if sum_zero_re:
        loc_set = set(loc_ids)
        for p in G.nodes:
            sibs = [int(c) for c in G.successors(p) if int(c) in loc_set]
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

    # 2) non-pivot 생성 (coord 순서대로)
    alpha_map: dict[int, Any] = {}
    for c in loc_ids:
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

    # 4) list → vector 등록 (dims=coord)
    alpha_list = [alpha_map[int(c)] for c in loc_ids]
    alpha_vec  = at.stack([at.as_tensor_variable(a) for a in alpha_list])
    pm.Deterministic(f"alpha_{data_type}", alpha_vec, dims=(id_dim,))

    sd[f'alpha_{data_type}'] = alpha_vec
    return alpha_vec

def build_fixed_effects_matrix(data_type: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    input_data_dt: pd.DataFrame = sd[f"input_data_{data_type}"].copy()

    # 1) X 구성
    keep = [c for c in input_data_dt.columns if c.startswith('x_')]
    X = input_data_dt[keep].copy()

    # x_sex 추가 (SEX_ID2NAME/SEX_NAME2VAL이 이미 정의돼 있다고 가정)
    SEX_ID2VAL = {id_: SEX_NAME2VAL[name] for id_, name in SEX_ID2NAME.items()}
    if 'sex_id' not in input_data_dt.columns:
        raise ValueError("input_data must contain 'sex_id' to build x_sex.")
    X['x_sex'] = [SEX_ID2VAL[int(row['sex_id'])] for _, row in input_data_dt.iterrows()]
    X = X.astype(float)

    # 2) ESS 가중치 확인
    if 'effective_sample_size' not in input_data_dt.columns:
        raise ValueError("'effective_sample_size' 컬럼이 필요합니다.")
    w = input_data_dt['effective_sample_size'].astype(float)
    if w.isna().any():
        raise ValueError("'effective_sample_size'에 NA 값이 있습니다.")
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

    # 6) covariate coord 등록
    cov_dim = f"fe_eff_name_{data_type}"
    cols = X_std.columns.tolist()
    pm_model.add_coord(cov_dim, cols, mutable=False)

    # 공유 저장
    sd[f'X_{data_type}']           = X_std
    sd[f'X_centering_{data_type}'] = X_centering
    sd[f'X_scaling_{data_type}']   = X_scaling
    return X_std, X_centering, X_scaling

def build_beta(data_type: str, X: pd.DataFrame):
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data
    params_dt = sd['parameters'][data_type]
    fe_specs  = (params_dt.get('fixed_effects') or {})

    cov_dim = f"fe_eff_name_{data_type}"
    pm_model.add_coord(cov_dim, X.columns.tolist(), mutable=False)
    cols = list(pm_model.coords[cov_dim])
        
    beta= []

    for effect in cols:
        name = f'beta_{data_type}_{effect}'
        spec = fe_specs.get(effect)

        if spec is None:
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
            beta.append(pm.Deterministic(name, -half) if sign == 'negative' else half)

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

    # 벡터화 + coord 연결
    beta_vec = at.stack([at.as_tensor_variable(b) for b in beta])
    pm.Deterministic(f"beta_{data_type}", beta_vec, dims=(cov_dim,))

    sd[f'beta_{data_type}']  = beta_vec
    return beta_vec

def mean_covariate_model(data_type: str, mu: at.TensorVariable):
    # -------- Random effects --------
    U, _ = build_random_effects_matrix(data_type)   # sd[f"U_{dt}"] 저장
    build_sigma_alpha(data_type)                    # sd[f"sigma_alpha_{dt}"] 저장
    alpha = build_alpha(data_type)              # Deterministic f"alpha_{dt}" (vector) 반환

    # -------- Fixed effects --------
    X, _, _ = build_fixed_effects_matrix(data_type) # sd[f"X_{dt}"] 저장
    beta = build_beta(data_type, X)             # Deterministic f"beta_{dt}" (vector) 반환

    n_obs = U.shape[0]

    # RE term
    if alpha is None or U.shape[1] == 0:
        rand_term = at.zeros((n_obs,))
    else:
        rand_term = at.dot(U.values, alpha)

    # FE term
    if beta is None or X.shape[1] == 0:
        fix_term = at.zeros((n_obs,))
    else:
        fix_term = at.dot(X.values, beta)

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
