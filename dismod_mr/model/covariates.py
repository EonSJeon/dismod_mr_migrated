import numpy as np
import pandas as pd
import pymc as pm
import networkx as nx
from typing import Dict, List, Tuple, Any
import pytensor.tensor as at
import warnings

SEX_VALUE = {'male': .5, 'total': 0., 'female': -.5}



def MyTruncatedNormal(name, mu, sigma, lower, upper, initval):
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



def build_sigma_alpha(
    data_type: str,
    parameters: Dict[str, Any],
    max_depth: int = 5
) -> List[Any]:
    """
    Generate hierarchical sigma_alpha priors via MyTruncatedNormal,
    preserving the original defaults from x_build_sigma_alpha.
    """
    sigma_alpha: List[Any] = []
    re_specs = parameters.get('random_effects', {})

    for i in range(max_depth):
        name = f'sigma_alpha_{data_type}_{i}'
        spec = re_specs.get(name)

        if spec:
            # 사용자 지정 하이퍼 prior
            mu = float(spec['mu'])
            s0 = max(float(spec['sigma']), 1e-3)
            lb = min(mu, spec['lower'])
            ub = max(mu, spec['upper'])
            init = float(spec['mu'])
        else:
            # 원래의 기본값 유지
            mu = 0.05
            s0 = 0.03
            lb = 0.05
            ub = 0.5
            init = 0.1

        sigma_alpha.append(
            MyTruncatedNormal(
                name=name,
                mu=mu,
                sigma=s0,
                lower=lb,
                upper=ub,
                initval=init
            )
        )

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
    랜덤 효과 계수 alpha, 상수 sigma 리스트, sum-to-zero potential 리스트 생성

    :param data_type: 파라미터 종류 문자열 (예: 'p', 'i' 등)
    :param U: 각 노드별 디자인 행렬 (DataFrame, 컬럼 이름이 노드명)
    :param sigma_alpha: level별 랜덤 효과 표준편차 RV 리스트
    :param parameters: user-specified prior 설정 dict
    :param zero_re: sum-to-zero 제약 적용 여부
    :param hierarchy: area hierarchy (DiGraph)
    :returns: (alpha 리스트, const_alpha_sigma 리스트, alpha_potentials 리스트)
    """
    alpha: List[Any] = []
    const_alpha_sigma: List[float] = []
    alpha_potentials: List[Any] = []

    # U에 컬럼이 없으면 바로 반환
    if U.shape[1] == 0:
        return alpha, const_alpha_sigma, alpha_potentials

    # 1) 각 컬럼(node)에 대응하는 sigma_alpha[level] 값 추출
    sigma_list = [
        sigma_alpha[hierarchy.nodes[c]['level']]
        for c in U.columns
    ]
    print(sigma_list)

    # 2) 각 노드마다 alpha RV 정의
    for col, sigma in zip(U.columns, sigma_list):
        name = f'alpha_{data_type}_{col}'
        spec = parameters.get('random_effects', {}).get(col)

        if spec:
            # 사용자 지정 prior이 있으면 그에 맞춰 분포 생성
            dist = spec['dist']
            if dist == 'Normal':
                mu0, s0 = float(spec['mu']), float(spec['sigma'])
                rv = pm.Normal(name, mu=mu0, sigma=s0, initval=0.0)
            elif dist == 'TruncatedNormal':
                mu0 = float(spec['mu'])
                s0  = max(float(spec['sigma']), 1e-3)
                lb, ub = spec['lower'], spec['upper']
                rv = MyTruncatedNormal(
                    name=name,
                    mu=mu0,
                    sigma=s0,
                    lower=lb,
                    upper=ub,
                    initval=0.0
                )
            elif dist == 'Constant':
                # 상수 prior인 경우 float로 처리
                rv = float(spec['mu'])
            else:
                raise ValueError(f"Unknown dist {dist} for {name}")
        else:
            # 기본 Normal(0, sigma_alpha[level]) prior
            rv = pm.Normal(name, mu=0.0, sigma=sigma, initval=0.0)

        alpha.append(rv)

        # Constant prior인 경우 상수 sigma 기록, 아니면 NaN
        const_alpha_sigma.append(
            float(spec.get('sigma', np.nan))
            if spec and spec.get('dist') == 'Constant'
            else np.nan
        )

    # 3) sum-to-zero 제약 (zero_re=True) 처리
    if zero_re:
        idx_map = {c: i for i, c in enumerate(U.columns)}
        for parent in hierarchy.nodes:
            # 자식 노드 중 U.columns에 있는 것만 필터
            children = [
                c for c in hierarchy.successors(parent)
                if c in idx_map
            ]
            # 형제가 2개 이상일 때만 sum-to-zero 제약 적용
            if len(children) < 2:
                continue

            # 첫 번째 자식 인덱스
            i0 = idx_map[children[0]]
            spec0 = parameters.get('random_effects', {}).get(children[0])
            # 첫 자식이 Constant prior이면 건너뛰기
            if spec0 and spec0.get('dist') == 'Constant':
                continue

            # 나머지 형제들 인덱스
            sibs = [idx_map[c] for c in children[1:]]
            # alpha[i0] = - sum(alpha[sibs])
            det = pm.Deterministic(
                f'alpha_det_{data_type}_{i0}',
                -sum(alpha[i] for i in sibs)
            )
            old = alpha[i0]
            alpha[i0] = det

            # 기존 stochastic이면 그 logp를 Potential로 보존
            if isinstance(old, pm.Distribution):
                alpha_potentials.append(
                    pm.Potential(
                        f'alpha_pot_{data_type}_{children[0]}',
                        old.logp(det)
                    )
                )

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
        data_type=data_type,
        U=U,
        sigma_alpha=sigma_alpha,
        parameters=parameters,
        zero_re=zero_re,
        hierarchy=model.hierarchy
    )


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
                beta.append(
                    MyTruncatedNormal(
                        name=name,
                        mu=float(spec['mu']),
                        sigma=max(float(spec['sigma']), 1e-3),
                        lower=float(spec['lower']),
                        upper=float(spec['upper']),
                        initval=float(spec['mu'])
                    )
                )
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



import numpy as np
import pandas as pd
import pymc as pm
from typing import Any, Dict


def dispersion_covariate_model(
    data_type: str,
    input_data: pd.DataFrame,
    delta_lb: float,
    delta_ub: float,
) -> Dict[str, Any]:
    """
    Generate dispersion (delta) covariate model in PyMC 5.3 style.

    Parameters
    ----------
    data_type : str
        모델 접두어 (예: 'p', 'i', 'r' 등)
    input_data : pd.DataFrame
        공변량(z_*) 컬럼이 포함된 DataFrame
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
    # ─── 0) 현재 PyMC 모델 컨텍스트를 가져와야 합니다 ─────────────────────────
    model = pm.modelcontext(None)
    assert model is not None, "dispersion_covariate_model must be called within a PyMC model"

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
        model.add_coord("covariate", Z.columns.tolist(), mutable=False)

        # (나) zeta ~ Normal(0, 0.25) 벡터, 길이 = len(Z.columns)
        zeta = pm.Normal(
            f"zeta_{data_type}",
            mu=0.0,
            sigma=0.25,
            dims=("covariate",),
            initval=np.zeros(len(Z.columns)),
        )

        # (다) “obs_dim” 차원(coord) 등록 (관측 개수만큼)
        model.add_coord("obs_dim", np.arange(len(input_data)), mutable=False)

        # (라) delta = exp(eta + Z.values @ zeta) 를 Deterministic으로 등록
        delta = pm.Deterministic(
            f"delta_{data_type}",
            pm.math.exp(eta + pm.math.dot(Z.values, zeta)),
            dims=("obs_dim",),
        )

        return {
            "eta":   [eta],
            "Z":      Z,
            "zeta": [zeta],
            "delta": [delta],
        }

    # ─── 5) Z가 없을 때 ────────────────────────────────────────────────────
    else:
        # (가) “obs_dim” 차원(coord) 등록
        model.add_coord("obs_dim", np.arange(len(input_data)), mutable=False)

        # (나) delta = exp(eta) * ones(len(input_data)) 형태로 생성
        const_delta = pm.Deterministic(
            f"delta_{data_type}",
            pm.math.exp(eta) * np.ones(len(input_data)),
            dims=("obs_dim",),
        )

        return {
            "eta":   [eta],
            "Z":      Z,            # 빈 DataFrame일 수도 있음
            "zeta": [],             # 공변량이 없으므로 빈 리스트
            "delta": [const_delta],
        }


import numpy as np
import networkx as nx
import pandas as pd

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
    vars: dict[str, any],
    lower: float,
    upper: float
) -> np.ndarray:
    """
    Generate posterior-predictive draws for a specific (area, sex, year).

    This version expects that you've already done something like
        with pm.Model() as _:
            … build the model …
            idata = pm.sample(…)
        model.idata = idata

    and that each RV (mu_age, alpha_*, beta_*, etc.) is recorded in idata.posterior.

    Returns
    -------
    preds : np.ndarray
        An array of shape (n_samples, n_ages) containing posterior draws 
        of the predicted rate for every age, clipped to [lower, upper].
    """

    # 1) Grab the InferenceData object:
    if not hasattr(model, "idata"):
        raise AssertionError("`model.idata` not found.  You must call pm.sample() inside the model and store the result as model.idata.")

    idata = model.idata

    # 2) Extract posterior draws of mu_age:
    mu_var = vars["mu_age"]       # this is the TensorVariable used for mu_age
    mu_name = mu_var.name         # e.g. "value_constrained_mu_age_p"
    if mu_name not in idata.posterior:
        raise KeyError(f"`{mu_name}` not found in idata.posterior.  Did you store the model trace in `model.idata`?")
    # idata.posterior[mu_name].values  has shape (n_chain, n_draw, n_ages)
    arr = idata.posterior[mu_name].values
    n_chain, n_draw, n_ages = arr.shape
    mu_trace = arr.reshape((n_chain * n_draw, n_ages))
    n_samples = mu_trace.shape[0]

    # 3) Build alpha_trace (random effects):
    #    If vars["alpha"] is present, it is a list of alpha-nodes;
    #    we look up each node’s posterior draws by name.  If a node was a constant (no trace),
    #    we simulate n_samples normal draws with the given “const_alpha_sigma” precision.
    alpha_trace = np.empty((n_samples, 0))
    if "alpha" in vars and isinstance(vars["alpha"], list) and vars["alpha"]:
        traces = []
        for alpha_node, sigma_const in zip(vars["alpha"], vars["const_alpha_sigma"]):
            name_alpha = alpha_node.name
            if name_alpha in idata.posterior:
                arr_a = idata.posterior[name_alpha].values  # (chain, draw)
                traces.append(arr_a.reshape(n_chain * n_draw))
            else:
                # treat alpha_node as a constant → simulate draws from Normal(loc=alpha_node, var=1/sigma_const)
                sig = max(sigma_const, 1e-9)
                loc = float(alpha_node)
                draws = np.random.normal(loc=loc, scale=1.0 / np.sqrt(sig), size=n_samples)
                traces.append(draws)
        alpha_trace = np.column_stack(traces)  # shape = (n_samples, len(vars["alpha"]))

    # 4) Build beta_trace (fixed effects), same logic as alpha:
    beta_trace = np.empty((n_samples, 0))
    if "beta" in vars and isinstance(vars["beta"], list) and vars["beta"]:
        traces = []
        for beta_node, sigma_const in zip(vars["beta"], vars["const_beta_sigma"]):
            name_beta = beta_node.name
            if name_beta in idata.posterior:
                arr_b = idata.posterior[name_beta].values  # (chain, draw)
                traces.append(arr_b.reshape(n_chain * n_draw))
            else:
                sig = max(sigma_const, 1e-9)
                loc = float(beta_node)
                draws = np.random.normal(loc=loc, scale=1.0 / np.sqrt(sig), size=n_samples)
                traces.append(draws)
        beta_trace = np.column_stack(traces)  # shape = (n_samples, len(vars["beta"]))

    # 5) Identify all leaf‐nodes under the requested “area”:
    leaves = [
        n for n in nx.bfs_tree(model.hierarchy, area) 
        if model.hierarchy.out_degree(n) == 0
    ]
    if not leaves:
        leaves = [area]

    # 6) Build a DataFrame of covariates/pop for each (area, sex, year):
    #    We assume `model.output_template` is a DataFrame with columns [area, sex, year, pop, ... covariates ...].
    output = model.output_template.copy()
    grp = output.groupby(["area", "sex", "year"], as_index=False).mean().set_index(["area", "sex", "year"])

    # 7) Prepare X_df (centered covariates) for the target (leaf, sex, year):
    if "X" in vars and not vars["X"].empty:
        X_df = grp.filter(vars["X"].columns, axis=1).copy()
        # set the “x_sex” column to the coded value for this sex:
        if "x_sex" in X_df.columns:
            # Here you need a mapping like SEX_VALUE = {"male": -0.5, "female": +0.5}
            X_df["x_sex"] = SEX_VALUE[sex]
        # subtract the shifts:
        X_df = X_df - vars["X_shift"]
    else:
        X_df = pd.DataFrame(index=grp.index)

    # 8) Prepare an all‐zero U_row Series; we’ll refill it for each leaf:
    if "U" in vars and not vars["U"].empty:
        U_cols = vars["U"].columns
        U_row = pd.Series(0.0, index=U_cols)
    else:
        U_row = pd.Series(dtype=float)

    # 9) Loop over each leaf to build “cov_shift” (shape = (n_samples,)):
    cov_shift = np.zeros(n_samples)
    total_weight = 0.0

    for leaf in leaves:
        # 9a) Build U‐indicator: 1 if this leaf is in that random‐effect, else 0,
        #     minus any “shift” for that area:
        U_row[:] = 0.0
        path = nx.shortest_path(model.hierarchy, root_area, leaf)
        for node in path[1:]:
            if node in U_row.index:
                U_row[node] = 1.0 - vars["U_shift"].get(node, 0.0)
        # Now compute α · U_row → (n_samples,) vector:
        if alpha_trace.size > 0:
            log_shift = alpha_trace.dot(U_row.values)  # shape=(n_samples,)
        else:
            log_shift = np.zeros(n_samples)

        # 9b) Add fixed‐effect contribution, if any:
        if beta_trace.size and (leaf, sex, year) in X_df.index:
            x_vals = X_df.loc[(leaf, sex, year)].values   # length = #covariates
            log_shift = log_shift + beta_trace.dot(x_vals)

        # 9c) Weight by population (or just average if not population_weighted):
        pop = float(grp.at[(leaf, sex, year), "pop"])
        if population_weighted:
            cov_shift += np.exp(log_shift) * pop
            total_weight += pop
        else:
            cov_shift += log_shift
            total_weight += 1.0

    # 10) Finish aggregation across leaves:
    if population_weighted:
        cov_shift = cov_shift / total_weight     # shape = (n_samples,)
    else:
        # if not population‐weighting, we took the sum of log_shift; divide by #leaves:
        cov_shift = np.exp(cov_shift / total_weight)

    # 11) Combine with baseline mu_age draws and clip:
    #     mu_trace has shape (n_samples, n_ages), cov_shift[:, None] has shape (n_samples, 1)
    preds = mu_trace * cov_shift[:, None]         # shape = (n_samples, n_ages)
    return np.clip(preds, lower, upper)
