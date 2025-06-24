import pymc as pm
import numpy as np
import networkx as nx
import dismod_mr

import time
import pymc as pm
import arviz as az
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def asr(
    mr_model: "dismod_mr.data.MRModel",
    pm_model: pm.Model,
    data_type: str,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    cores: int = 4,
    target_accept: float = 0.9995,
    max_treedepth: int = 10,
    use_advi: bool = False,         # ← ADVI 모드 on/off
    use_metropolis: bool = False,   # ← Metropolis 모드 on/off
    vi_iters: int = 20000,
    vi_lr: float = 1e-3,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], pm.backends.arviz.InferenceData]:
    """
    Fit one age‐specific rate model via PyMC v5.3 (ADVI, Metropolis, or NUTS).

    - use_advi=True  → pm.fit() → approx.sample()
    - use_metropolis=True → pm.Metropolis() via pm.sample(step=...)
    - else → default NUTS with target_accept & max_treedepth
    """
    # 1) 입력 확인
    assert pm_model is not None, "asr는 pm_model이 설정된 후 호출되어야 합니다."
    if not hasattr(mr_model, "vars") or data_type not in mr_model.vars:
        raise AssertionError(
            f"mr_model.vars[{data_type!r}] 가 없습니다. 먼저 `setup_model(..., rate_type=data_type)` 호출하세요."
        )
    vars_dict = mr_model.vars[data_type]

    # 2) MAP 초기화
    with pm_model:
        if verbose:
            logger.info("  ▶ pm.find_MAP() 수행 중...")
        map_estimate = pm.find_MAP()

    # 3) Posterior approximation / sampling
    t_start = time.time()
    with pm_model:
        if use_advi:
            if verbose:
                logger.info("  ▶ ADVI 수행 중...")
            approx = pm.fit(
                n=vi_iters,
                method="advi",
                obj_optimizer=pm.adam(learning_rate=vi_lr),
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
            )
            idata = approx.sample(draws=draws)

        elif use_metropolis:
            if verbose:
                logger.info("  ▶ Metropolis 샘플링 수행 중...")
            step = pm.Metropolis()
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step,
                start=map_estimate,
                return_inferencedata=True,
                progressbar=verbose,
            )

        else:
            if verbose:
                logger.info("  ▶ NUTS 샘플링 수행 중...")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                start=map_estimate,
                target_accept=target_accept,
                nuts={"max_treedepth": max_treedepth},
                return_inferencedata=True,
                progressbar=verbose,
            )
    t_end = time.time()
    wall_time = t_end - t_start
    if verbose:
        logger.info(f"[asr] 전체 소요 시간: {wall_time:.1f}초")

    # 4) MRModel에 저장
    mr_model.idata = idata
    mr_model.map_estimate = map_estimate
    mr_model.wall_time = wall_time

    # 5) 반환
    if use_advi:
        return map_estimate, idata, approx
    else:
        return map_estimate, idata


def consistent(
    model_data,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 1,
    target_accept: float = 0.9,
    verbose: bool = False
):
    """
    PyMC 5.3 스타일로 “모든 rate_type(예: 'i','r','f','p','pf',…)”을 동시에 Fitting 합니다.

    모델 구성(연령별 spline, 공변량, 랜덤효과, Likelihood 등)은 이미
    `model_data.pm_model` 내부에서 완료되어 있다고 가정합니다.
    그리고 각 rate_type별 변수 사전은 `model_data.vars[t]`에 보관되어 있어야 합니다.

    Parameters
    ----------
    model_data : object
        - age_specific_rate(...) 등을 통해 미리 생성된 객체. 
        - 반드시 `model_data.pm_model` 속성에 `pm.Model()` 인스턴스가 있어야 합니다.
        - 또한 `model_data.vars`는 rate_type별 딕셔너리여야 합니다. 
          예: `model_data.vars['p']['gamma']`, `model_data.vars['p']['mu_age']` 등.

    draws : int
        최종 posterior 샘플 수 (default=2000)
    tune : int
        적응(tuning) 단계 샘플 수 (default=1000)
    chains : int
        MCMC 체인 수 (default=2)
    cores : int
        샘플링 시 사용할 CPU 코어 수 (default=1)
    target_accept : float
        NUTS 목표 수락율 (default=0.9)
    verbose : bool
        True면 진행 상황을 로깅/출력

    Returns
    -------
    map_estimate : dict
        MAP 추정치 (pm.find_MAP()로 얻은 파라미터 dict)
    trace : arviz.InferenceData
        pm.sample() 결과 InferenceData
    """

    # 1) 사전 조건 검사
    if not hasattr(model_data, "pm_model") or not isinstance(model_data.pm_model, pm.Model):
        raise AssertionError("`model_data.pm_model` 에 PyMC Model 객체가 없습니다. 먼저 age_specific_rate(...)를 호출하여 pm_model을 설정해야 합니다.")

    if not hasattr(model_data, "vars") or not isinstance(model_data.vars, dict):
        raise AssertionError("`model_data.vars` 가 정의되어 있지 않거나 형식이 잘못되었습니다. 연령별 서브모델 변수 사전이 필요합니다.")

    # (옵션) param_types 리스트 확인
    param_types = list(model_data.vars.keys())
    if verbose:
        logger.info(f"[consistent] fitting for rate types: {param_types}")

    pm_model: pm.Model = model_data.pm_model

    # 2) MAP 추정 (pm.find_MAP)
    start_time = time.time()
    with pm_model:
        if verbose:
            logger.info("  ▶ MAP estimate 계산 중 (`pm.find_MAP()` 호출)")
        map_estimate = pm.find_MAP()
        if verbose:
            logger.info("    ▶ MAP 추정치 (몇 가지 주요 파라미터만 일부 출력)")
            # 가령 gamma_p_0, delta_p 등 일부 파라미터만 출력해 보고 싶으면 이렇게 하면 됩니다.
            for k, v in map_estimate.items():
                try:
                    val = v.eval()
                except Exception:
                    val = v
                logger.info(f"      {k}: {val}")

    # 3) NUTS 샘플링
    with pm_model:
        if verbose:
            logger.info("  ▶ NUTS 샘플링 시작 (`pm.sample` 호출)")
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            start=map_estimate,
            target_accept=target_accept,
            progressbar=verbose,
            return_inferencedata=True
        )
        if verbose:
            logger.info("    ▶ 샘플링 완료")

    wall_time = time.time() - start_time
    if verbose:
        logger.info(f"[consistent] 전체 소요 시간: {wall_time:.1f} 초")

    # 4) 결과 저장
    model_data.map_estimate = map_estimate
    model_data.trace = trace
    model_data.wall_time = wall_time

    return map_estimate, trace


import numpy as np
import logging

# ─── 1) MKL 쓰레드 제어 (선택 사항) ────────────────────────────────────
#    mkl이 설치되어 있으면, BLAS/NumPy 연산 시 사용하는 쓰레드를 1개로 제한합니다.
try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass


# ─── 2) Median Absolute Relative Error 출력 함수 ─────────────────────────
def print_mare(vars: dict) -> None:
    """
    Print the Median Absolute Relative Error (MARE) between p_obs and pi.

    Expects:
      - vars['p_obs'].value  : observed values (array‐like)
      - vars['pi'].value     : expected values (array‐like)

    If either key is missing, 아무 일도 하지 않습니다.
    """
    p_obs_node = vars.get('p_obs')
    pi_node     = vars.get('pi')

    if p_obs_node is None or pi_node is None:
        return

    # p_obs_node.value 와 pi_node.value 에서 NumPy 배열을 가져오려고 시도
    try:
        obs = np.atleast_1d(p_obs_node.value)
        exp = np.atleast_1d(pi_node.value)
    except Exception:
        # .value 속성이 없으면 그냥 NumPy array로 변환
        obs = np.asarray(p_obs_node)
        exp = np.asarray(pi_node)

    # 0으로 나누는 상황 방지: exp가 0인 요소는 제외
    nonzero = exp != 0
    if not np.any(nonzero):
        print("mare: undefined (all expected values are zero)")
        return

    are = np.abs((obs[nonzero] - exp[nonzero]) / exp[nonzero])
    mare = np.median(are)

    print(f"mare: {mare:.2f}", flush=True)



# ─── 3) 간단한 로거 설정 ───────────────────────────────────────────────
#    Python 표준 logging 모듈을 사용하여 로그 레벨별 출력이 가능하도록 설정합니다.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.handlers.clear()
logger.addHandler(handler)


# ─── 4) 파라미터 타입 리스트 ───────────────────────────────────────────
param_types = ['i', 'r', 'f', 'p', 'pf', 'rr', 'smr', 'm_with', 'X']




def find_consistent_spline_initial_vals(vars: dict, method: str, tol: float, verbose: bool):
    """
    이전에는 mc.MAP(...).fit(...)을 사용하던 부분을 pm.find_MAP(...) 호출로 대체했습니다.
    각 단계별로 필요한 잠재변수들만 골라 MAP 추정치를 구한 뒤, test_value로 설정합니다.
    """
    # ── 1) 기본적으로 항상 고정하고 싶은 변수들 ───────────────────────────
    vars_to_fit = [vars.get('logit_C0')]
    for t in param_types:
        vars_to_fit += [
            vars[t].get('mu_age_derivative_potential'),
            vars[t].get('mu_sim'),
            vars[t].get('p_obs'),
            vars[t].get('parent_similarity'),
            vars[t].get('smooth_gamma'),
            vars[t].get('covariate_constraint'),
        ]
    # None인 항목은 제외
    vars_to_fit = [v for v in vars_to_fit if v is not None]

    # ── 2) gamma 개수를 rate_type 'i','r','f' 중에서 가장 큰 수로 결정 ───────
    max_knots = max(len(vars[t]['gamma']) for t in ['i', 'r', 'f'])
    for i in [max_knots]:
        if verbose:
            print(f"Fitting first {i} of {max_knots} spline knots...")
        # 각 rate_type별 gamma[0:i]를 추가
        knot_vars = []
        for t in ['i', 'r', 'f']:
            knot_vars += vars[t]['gamma'][:i]
        fit_vars = vars_to_fit + [k for k in knot_vars if k is not None]

        # pm.find_MAP 을 사용하여 MAP 추정치 계산
        map_vals = pm.find_MAP(vars=fit_vars, method=method, tol=tol)
        # 각 변수의 test_value를 MAP 추정치로 업데이트
        for v in fit_vars:
            name = getattr(v, 'name', None)
            if name and name in map_vals:
                try:
                    v.tag.test_value = map_vals[name]
                except Exception:
                    pass

        if verbose:
            from fit_posterior import inspect_vars
            print(inspect_vars({}, vars)[-10:])
        else:
            print(".", end=" ", flush=True)

def find_asr_initial_vals(vars: dict, method: str, tol: float, verbose: bool):
    """
    PyMC v5용 초기값 찾기 루틴:
    ── 스플라인 → 랜덤 효과 → 스플라인 → 고정 효과 → 스플라인 → 분산 순으로
       여러 번 pm.find_MAP을 호출해 test_value를 업데이트합니다.
    """
    for _ in range(3):
        # 1) 스플라인 매개변수 초기값
        find_spline_initial_vals(vars, method, tol, verbose)

        # 2) 랜덤 효과(alpha, sigma_alpha 등) 초기값
        find_re_initial_vals(vars, method, tol, verbose)

        # 3) 다시 스플라인 매개변수 초기값
        find_spline_initial_vals(vars, method, tol, verbose)

        # 4) 고정 효과(beta 등) 초기값
        find_fe_initial_vals(vars, method, tol, verbose)

        # 5) 다시 스플라인 매개변수 초기값
        find_spline_initial_vals(vars, method, tol, verbose)

        # 6) 분산(eta, zeta) 초기값
        find_dispersion_initial_vals(vars, method, tol, verbose)

        # 진행 표시
        logger.heartbeat()


def find_spline_initial_vals(vars: dict, method: str, tol: float, verbose: bool):
    """
    기존의 find_spline_initial_vals를 거의 그대로 두되,
    pm.find_MAP만 호출합니다.
    """
    base_vars = [
        vars.get('p_obs'),
        vars.get('pi_sim'),
        vars.get('parent_similarity'),
        vars.get('mu_sim'),
        vars.get('mu_age_derivative_potential'),
        vars.get('covariate_constraint'),
    ]
    base_vars = [v for v in base_vars if v is not None]

    for i, knot in enumerate(vars.get('gamma', []), start=1):
        if verbose:
            print(f"Fitting first {i} of {len(vars['gamma'])} spline knots...")
        fit_vars = base_vars + [knot]

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


def find_re_initial_vals(vars: dict, method: str, tol: float, verbose: bool):
    """
    계층적 랜덤 효과 초기화: pm.find_MAP 호출
    """
    if 'hierarchy' not in vars or 'U' not in vars:
        return

    col_map = {col: idx for idx, col in enumerate(vars['U'].columns)}

    # BFS 순회하며 부모-자식 그룹별 alpha 초기화
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

    # 마지막으로 sigma_alpha 초기화
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


def find_fe_initial_vals(vars: dict, method: str, tol: float, verbose: bool):
    """
    고정 효과(beta) 초기화: pm.find_MAP 호출
    """
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


def find_dispersion_initial_vals(vars: dict, method: str, tol: float, verbose: bool):
    """
    분산(eta, zeta) 초기화: pm.find_MAP 호출
    """
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


def setup_asr_step_methods(vars: dict, additional_rvs=None):
    """
    과거 mc.NormApprox, mc.AdaptiveMetropolis 등을 사용하던 부분은
    PyMC v5에서는 NUTS, Metropolis 등의 스텝 메서드를 직접 지정할 수 있습니다.
    예시로, 특별한 사전 계산 없이 기본 NUTS를 사용하도록 지정하는 패턴을 보여드립니다.
    """
    # 예: gamma나 alpha 같은 연속형 파라미터에 대해 NUTS를 쓰고 싶다면:
    for t in ['i', 'r', 'f']:
        for gamma_node in vars[t].get('gamma', []):
            if gamma_node is not None:
                pm.NUTS(vars=[gamma_node])  # 또는 원하는 step method로 교체
    # 필요하다면 추가 RVS에 대해 다른 스텝 함수를 지정할 수도 있습니다.
    # 예: additional_rvs에 discrete 변수가 들어오면 Metropolis 등을 설정

    # (참고) 실제로는 with pm.Model(): 내에서
    #        trace = pm.sample(...) 호출 시 step 인자로 이 step 메서드 리스트를 넘겨주면 됩니다.


    """
    Prepare a list of Metropolis step methods for Adaptive Metropolis sampling in PyMC v4+.

    Parameters
    ----------
    vars : dict
        A mapping of variable names to PyMC random variables (DistributionRV).
        Expected keys include:
          - 'beta': list of fixed-effect RVs
          - 'gamma': list of spline-coefficient RVs
          - 'alpha': list or array of hierarchy RVs
          - 'U': pandas DataFrame for hierarchy columns
          - 'hierarchy': networkx.Graph representing hierarchical structure
    additional_rvs : list, optional
        Additional RVs to include in their own Metropolis sampler.

    Returns
    -------
    steps : list
        A list of pm.Metropolis step-method instances for use in pm.sample(step=...).
    """
    steps = []
    if additional_rvs is None:
        additional_rvs = []

    # group fixed effects and spline coefficients
    fe_group = list(vars.get('beta', []))
    ap_group = list(vars.get('gamma', []))

    # chain-adjacent gamma pairs
    chain_pairs = [[ap_group[i], ap_group[i-1]]
                   for i in range(1, len(ap_group))]

    # prepare all grouping patterns
    groupings = chain_pairs + [fe_group, ap_group, fe_group + ap_group]

    # hierarchy-based groups
    if 'hierarchy' in vars and 'alpha' in vars and 'U' in vars:
        col_map = {key: i for i, key in enumerate(vars['U'].columns)}
        for node in vars['hierarchy'].nodes:
            group = []
            # collect alpha RVs along path from 'all' to node
            try:
                path = nx.shortest_path(vars['hierarchy'], 'all', node)
            except nx.NetworkXNoPath:
                continue
            for anc in path:
                if anc in vars['U'].columns:
                    rv = vars['alpha'][col_map[anc]]
                    group.append(rv)
            if group:
                groupings.append(group)

    # create a Metropolis sampler for each non-empty group
    for grp in groupings:
        if grp and all(getattr(r, 'owner', None) is not None for r in grp):
            steps.append(pm.Metropolis(vars=grp))

    # any additional RVs get their own sampler
    if additional_rvs:
        steps.append(pm.Metropolis(vars=additional_rvs))

    return steps