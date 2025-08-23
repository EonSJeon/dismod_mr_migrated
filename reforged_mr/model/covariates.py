import numpy as np
import pandas as pd
import pymc as pm
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at


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

def build_alpha(data_type: str) -> Tuple[List[Any], List[float]]:
    """
    Create per-node random effects alpha for U_{data_type}, enforcing sibling sum-to-zero
    without re-registering variables. Returns (alpha_list, const_alpha_sigma) aligned to U.columns.
    """
    pm_model = pm.modelcontext(None)
    sd = pm_model.shared_data

    G: nx.DiGraph      = sd['region_id_graph']
    U: pd.DataFrame    = sd[f'U_{data_type}']
    sigma_alpha        = sd[f'sigma_alpha_{data_type}']
    params_dt          = sd['parameters'][data_type]
    specs              = params_dt.get('random_effects', {}) or {}
    sum_zero_re        = bool(params_dt.get('sum_zero_re', params_dt.get('zero_re', True)))

    # 비어 있으면 바로 반환
    if U.shape[1] == 0:
        return [], []

    cols = list(U.columns)

    # ---- helpers -------------------------------------------------------------
    def _get_spec(c):
        # random_effects의 키가 str/int 섞일 수 있어 양쪽 조회
        return specs.get(c, specs.get(str(c), specs.get(int(c), None)))

    def _sigma_for(c):
        lvl = G.nodes[c]['level']
        return sigma_alpha[lvl]

    def _make_alpha_rv_or_const(c):
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
        # default
        return pm.Normal(name, mu=0.0, sigma=_sigma_for(c), initval=0.0)

    # ---- 1) pivot plan: 각 부모의 형제 중 합=0 제약 대상 식별 -------------------
    pivot_plan: dict[int, tuple[list[int], float]] = {}  # pivot -> (rest_free, const_sum)
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

    # ---- 2) RV 생성: 피벗은 건너뛰고 나머지만 한 번 생성 ------------------------
    alpha_map: dict[int, Any] = {}
    const_sigma_map: dict[int, float] = {}
    for c in cols:
        if c in pivot_nodes:
            const_sigma_map[c] = np.nan  # pivot은 나중에 Deterministic으로
            continue
        a = _make_alpha_rv_or_const(c)
        alpha_map[c] = a
        sp = _get_spec(c)
        const_sigma_map[c] = float(sp.get('sigma', np.nan)) if (sp and sp.get('dist') == 'Constant') else np.nan

    # ---- 3) pivot 정의: -(나머지 자유형제 합 + 상수합) --------------------------
    for pivot, (rest, const_sum) in pivot_plan.items():
        # rest 중 미생성된 노드가 있으면 생성
        for r in rest:
            if r not in alpha_map:
                alpha_map[r] = _make_alpha_rv_or_const(r)
                const_sigma_map[r] = np.nan
        sum_rest = sum(alpha_map[r] for r in rest) if rest else 0.0
        alpha_map[pivot] = pm.Deterministic(
            f'alpha_{data_type}_{pivot}',
            -(sum_rest + const_sum)
        )
        const_sigma_map[pivot] = np.nan

    # ---- 4) 리스트로 정렬 & 반환 ----------------------------------------------
    alpha_list = [alpha_map[c] for c in cols]
    const_alpha_sigma = [const_sigma_map.get(c, np.nan) for c in cols]
    sd[f'alpha_{data_type}'] = alpha_list
    sd[f'const_alpha_sigma_{data_type}'] = const_alpha_sigma
    return alpha_list, const_alpha_sigma

def mean_covariate_model(data_type: str, mu: at.TensorVariable): 
    # NOTE:
    # U_ref and X_centering have different functions. 
    # U_ref is 0/1 indicator vector to make U(ref) = 0, 
    # X_centering is a vector of mean of covariates for the centering literally.

    pm_model  = pm.modelcontext(None)
    sd        = pm_model.shared_data
    parameters = sd["parameters"]
    params_of_data_type = parameters[data_type]
    input_data_dt = sd[f"input_data_{data_type}"]

    # --------------------------- build random effects matrix ---------------------------   
    build_random_effects_matrix(data_type)
    build_sigma_alpha(data_type)
    alpha, const_alpha_sigma = build_alpha(data_type)

    # --------------------------- build covariate matrix ---------------------------   
    keep = [c for c in input_data_dt.columns if c.startswith('x_')]
    X = input_data_dt[keep].copy()
    X['x_sex'] = [SEX_VALUE[row['sex_id']] for _, row in input_data_dt.iterrows()]
    X = X.astype(float)

    # --- 2) 분석 가중치: effective_sample_size 필수
    if 'effective_sample_size' not in input_data_dt.columns:
        raise ValueError("'effective_sample_size' 컬럼이 필요합니다.")
    
    w = input_data_dt['effective_sample_size'].astype(float)
    
    if w.isna().any():
        raise ValueError("'effective_sample_size'에 NA 값이 있습니다. 모든 값이 유효해야 합니다.")

    w_sum = float(w.sum())

    # --- 3) 가중 평균(centering, data 기반)
    X_centering = X.mul(w, axis=0).sum(axis=0) / w_sum

    # --- 4) 가중 표준편차(분산=1이 되도록), 0 회피 바닥값
    def _wstd(col: pd.Series, w: pd.Series) -> float:
        m = (col * w).sum() / w.sum()
        var = (w * (col - m)**2).sum() / w.sum()  # population weighted variance
        return float(np.sqrt(var))

    X_scaling = X.apply(lambda c: max(_wstd(c, w), 1e-12))

    # --- 5) 표준화: (X - mean) / std
    X = (X - X_centering) / X_scaling

    beta = []
    const_beta_sigma = []
    for effect in X.columns:
        name = f'beta_{data_type}_{effect}'
        spec = params_of_data_type.get('fixed_effects', {}).get(effect)
        if spec:
            dist = spec['dist']
            if dist == 'Zero':
                beta.append(pm.Deterministic(name, at.as_tensor_variable(0.0)))
            elif dist == 'TruncatedNormal':
                beta.append(
                    MyTruncatedNormal(
                        name=name,
                        mu=float(spec['mu']),
                        sigma=max(float(spec['sigma']), 1e-3),
                        lower=float(spec['lower']),
                        upper=float(spec['upper'])
                    )
                )
            elif dist == 'HalfNormal':
                sign = spec.get('sign', 'positive')  # 기본은 양수
                if sign == 'negative':
                    half = pm.HalfNormal(
                        name + "_half",
                        sigma=max(float(spec.get('sigma', 1.0)), 1e-3),
                        initval=abs(spec.get('initval', 0.1))
                    )
                    beta.append(pm.Deterministic(name, -half))
                else:
                    beta.append(
                        pm.HalfNormal(
                            name,
                            sigma=max(float(spec.get('sigma', 1.0)), 1e-3),
                            initval=spec.get('initval', 0.1)
                        )
                    )
            else: # Normal
                beta.append(
                    pm.Normal(
                        name,
                        mu=spec.get('mu', 0),
                        sigma=spec.get('sigma', 1)
                    )
                )

            const_beta_sigma.append(spec.get('sigma') if dist == 'Constant' else np.nan)
        else:
            beta.append(pm.Normal(name, mu=0.0, sigma=1.0))
            const_beta_sigma.append(np.nan)


    n_obs = U.shape[0]

    if alpha:
        alpha_stack = pm.math.stack(alpha)
        rand_term   = pm.math.dot(U.values, alpha_stack)
    else:
        rand_term   = at.zeros((n_obs,))

    if beta:
        beta_stack = pm.math.stack(beta)
        fix_term   = pm.math.dot(X.values, beta_stack)
    else:
        fix_term   = at.zeros((n_obs,))

    pi = pm.Deterministic(
        f"pi_{data_type}",
        mu * pm.math.exp(rand_term + fix_term)
    )

    # --------------------------- 3) store shared data ---------------------------   

    pm_model.shared_data['X']                 = X
    pm_model.shared_data['X_centering']       = X_centering
    pm_model.shared_data['X_scaling']         = X_scaling
    pm_model.shared_data['beta']              = beta
    pm_model.shared_data['const_beta_sigma']  = const_beta_sigma


def dispersion_covariate_model(
    delta_lb: float,
    delta_ub: float,
) -> Dict[str, Any]:
    """
    Generate dispersion (delta) covariate model in PyMC 5.3 style.

    Parameters
    ----------
    delta_lb : float
        delta 하한 (양수)
    delta_ub : float
        delta 상한 (양수)

    Returns
    -------
    Dict[str, Any]
        - eta   : [Uniform RV on log(delta)]
        - Z     : DataFrame slice of z_* covariates (원본 DataFrame에서 복사본)
        - zeta  : [Normal RV vector]  # Z가 있을 때만 반환
        - delta : [Deterministic]      # exp(eta + Z @ zeta) 또는 exp(eta) * ones
    """

    # --------------------------- 1) initialize pm_model ---------------------------   
    pm_model = pm.modelcontext(None) # at reforged_mr/model/covariates/dispersion_covariate_model()


    # --------------------------- 2) extract shared data ---------------------------   
    data_type = pm_model.shared_data["data_type"]
    input_data = pm_model.shared_data["data"]

    # ─── 1) log(delta)의 하한/상한 계산 ──────────────────────────────────────
    lower = np.log(delta_lb)
    upper = np.log(delta_ub)

    # ─── 2) eta ~ Uniform(log(delta_lb), log(delta_ub)) ────────────────────
    eta = pm.Uniform(
        f"eta_{data_type}",
        lower=lower,
        upper=upper,
        initval=0.5 * (lower + upper),
        # dims 지정은 필요 없으므로 생략
    )

    # ─── 3) “z_” 로 시작하고 분산(std) > 0인 컬럼만 골라냄 ─────────────────────
    keep_cols = [
        c for c in input_data.columns
        if c.startswith("z_") and input_data[c].std() > 0
    ]
    Z = input_data[keep_cols].copy()

    # ─── 4) Z가 하나라도 있을 때 ───────────────────────────────────────────
    if len(Z.columns) > 0:
        # (가) “covariate” 차원(coord) 먼저 등록
        pm_model.add_coord("covariate", Z.columns.tolist(), mutable=False)

        # (나) zeta ~ Normal(0, 0.25) 벡터, 길이 = len(Z.columns)
        zeta = pm.Normal(
            f"zeta_{data_type}",
            mu=0.0,
            sigma=0.25,
            dims=("covariate",),
            initval=np.zeros(len(Z.columns)),
        )

        # (다) “obs_dim” 차원(coord) 등록 (관측 개수만큼)
        pm_model.add_coord("obs_dim", np.arange(len(input_data)), mutable=False)

        # (라) delta = exp(eta + Z.values @ zeta) 를 Deterministic으로 등록
        delta = pm.Deterministic(
            f"delta_{data_type}",
            pm.math.exp(eta + pm.math.dot(Z.values, zeta)),
            dims=("obs_dim",),
        )

        return delta
    # ─── 5) Z가 없을 때 ────────────────────────────────────────────────────
    else:
        # (가) “obs_dim” 차원(coord) 등록
        pm_model.add_coord("obs_dim", np.arange(len(input_data)), mutable=False)

        # (나) delta = exp(eta) * ones(len(input_data)) 형태로 생성
        const_delta = pm.Deterministic(
            f"delta_{data_type}",
            pm.math.exp(eta) * np.ones(len(input_data)),
            dims=("obs_dim",),
        )

        return const_delta