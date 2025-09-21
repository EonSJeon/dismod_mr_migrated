import numpy as np
import pymc as pm
import pytensor.tensor as at
import warnings
import pandas as pd

LARGE_N_FOR_PRED = 1e6
EPS = 1e-9

def binom(data_type, obs_data, pi) -> None:
    p = obs_data['value'].to_numpy(dtype=float)
    n = obs_data['effective_sample_size'].to_numpy(dtype=int)

    assert np.all(p >= 0.0), "observed values must be non-negative"
    assert np.all(p <= 1.0), "observed values must be <= 1"
    assert np.all(n >= 0),   "effective sample size must be non-negative"

    # --- observed counts (effective_cases) ---
    effective_cases = np.clip(np.rint(p * n), 0, n).astype(int)
    n_int = n.astype(int)

    # --- observed likelihood ---
    p_obs = at.clip(pi, EPS, 1.0 - EPS)
    pm.Binomial(
        name=f"p_obs_{data_type}",
        n=n_int,
        p=p_obs,
        observed=effective_cases,
    )

    # --- predictive with n==0 handling & warning ---
    zero_mask = (n_int == 0)
    if np.any(zero_mask):
        idx_list = obs_data.index.to_numpy()[zero_mask]
        warnings.warn(
            f"[binom] effective_cases == 0 for {zero_mask.sum()} rows; "
            f"using n_pred={LARGE_N_FOR_PRED} for predictive rate. indices={idx_list}",
            RuntimeWarning,
        )

    n_pred = n_int.copy()
    n_pred[zero_mask] = LARGE_N_FOR_PRED

    count_pred = pm.Binomial(name=f"p_count_{data_type}", n=n_pred, p=p_obs)
    pm.Deterministic(name=f"p_pred_{data_type}", var=count_pred / n_pred)

def poisson(data_type, obs_data, pi) -> None:
    # prevalence in [0,1], ESS >= 0
    p = obs_data['value'].to_numpy(dtype=float)
    n = obs_data['effective_sample_size'].to_numpy(dtype=int)

    assert np.all(p >= 0), "observed values must be non-negative"
    assert np.all(p <= 1.0), "observed values must be <= 1"
    assert np.all(n >= 0), "effective sample size must be non-negative"

    # counts = Binomial(n,p) ~ Poisson(np) 근사
    obs_counts = np.clip(np.rint(p * n), 0, n).astype(int)
    n_int = n.astype(int)

    pm.Poisson(
        name=f"p_obs_{data_type}",
        mu=pi * n_int,
        observed=obs_counts,
    )

    # 예측용 n이 0인 경우 경고
    zero_mask = (n_int == 0)
    if np.any(zero_mask):
        idx_list = obs_data.index.to_numpy()[zero_mask]
        warnings.warn(
            f"[poisson] effective_cases == 0 for {zero_mask.sum()} rows; "
            f"using large n_pred=1e6 for predictive rate. indices={idx_list}",
            RuntimeWarning,
        )

    n_pred = n_int.copy()
    n_pred[zero_mask] = LARGE_N_FOR_PRED

    count_pred = pm.Poisson(name=f"p_count_{data_type}", mu=pi * n_pred)
    pm.Deterministic(name=f"p_pred_{data_type}", var=count_pred / n_pred)


def neg_binom(
    name,
    obs_or_p,              # DataFrame (옛버전) 또는 p-array (신버전)
    pi,                    # TensorVariable 또는 array-like
    delta,                 # TensorVariable 또는 스칼라/길이1 리스트
    n: np.ndarray = None   # 신버전 경로일 때만 필요
) -> dict:
    """
    Backward-compatible NegativeBinomial likelihood:
      - 호출 1) neg_binom(name, obs_data_df, pi, delta)
      - 호출 2) neg_binom(name, p_array,      pi, delta, n_array)

    옛버전 동작(검증/클리핑/경고/빈 케이스 가드)을 그대로 유지.
    Returns {"p_obs": None, "p_pred": Deterministic}
    """
    assert pm.modelcontext(None) is not None, "neg_binom() must be called within a PyMC model"

    # --- delta가 [x] 형태면 스칼라로 풀기 ---
    if isinstance(delta, list) and len(delta) == 1:
        delta = delta[0]

    # --- 입력 분기: DF 또는 배열 ---
    if isinstance(obs_or_p, pd.DataFrame):
        df = obs_or_p
        if not {"value", "effective_sample_size"}.issubset(df.columns):
            raise ValueError("obs_data must have columns {'value','effective_sample_size'}")
        p = df["value"].to_numpy(dtype=float)
        n_arr = df["effective_sample_size"].to_numpy(dtype=float)
        idx_for_warn = df.index.to_numpy()
    else:
        # array 경로
        if n is None:
            raise TypeError("When passing arrays, you must provide both p and n.")
        p = np.asarray(obs_or_p, dtype=float)
        n_arr = np.asarray(n, dtype=float)
        idx_for_warn = np.arange(p.shape[0])

    # --- 검증 (옛버전 정책) ---
    assert np.all(p >= 0.0), "observed values must be non-negative"
    assert np.all(p <= 1.0), "observed values must be <= 1"
    assert np.all(n_arr >= 0.0), "effective sample size must be non-negative"

    # --- 카운트 생성: rint + clip + int64 ---
    obs_counts = np.clip(np.rint(p * n_arr), 0, n_arr).astype(np.int64)

    n_float = n_arr.copy()
    nonzero_idx = np.where(n_float > 0.0)[0]

    # --- mu/alpha 텐서 ---
    mu_obs_all = pi * n_float + 1e-9
    alpha_all  = delta

    # --- 관측 우도 ---
    if nonzero_idx.size > 0:
        mu_obs = at.take(mu_obs_all, nonzero_idx)
        if hasattr(alpha_all, "shape"):
            alpha_obs = at.take(alpha_all, nonzero_idx)
        else:
            alpha_obs = alpha_all

        nb_dist  = pm.NegativeBinomial.dist(mu=mu_obs, alpha=alpha_obs)
        logp_val = pm.logp(nb_dist, obs_counts[nonzero_idx])
        pm.Potential(name=f"p_obs_{name}", var=at.sum(logp_val))
    else:
        pm.Potential(name=f"p_obs_{name}", var=at.as_tensor_variable(0.0))

    # --- predictive: n==0 처리 ---
    LARGE_N_FOR_PRED = int(1e9)
    zero_mask = (n_float == 0.0)
    if np.any(zero_mask):
        warnings.warn(
            f"[neg_binom] effective_sample_size == 0 for {int(zero_mask.sum())} rows; "
            f"using n_pred={LARGE_N_FOR_PRED} for predictive rate. "
            f"indices={(idx_for_warn[zero_mask]).tolist()}",
            RuntimeWarning,
        )

    n_pred = n_float.copy()
    n_pred[zero_mask] = float(LARGE_N_FOR_PRED)
    mu_pred = pi * n_pred + 1e-9

    p_pred = pm.Deterministic(name=f"p_pred_{name}", var=mu_pred / n_pred)

    return {"p_obs": None, "p_pred": p_pred}


def neg_binom_x(data_type, obs_data, pi, delta) -> None:

    p = obs_data['value'].to_numpy(dtype=float)
    n = obs_data['effective_sample_size'].to_numpy(dtype=float)

    assert np.all(p >= 0.0), "observed values must be non-negative"
    assert np.all(p <= 1.0), "observed values must be <= 1"
    assert np.all(n >= 0),   "effective sample size must be non-negative"

    # ---- observed counts (effective_cases) ----
    obs_counts = np.clip(np.rint(p * n), 0, n).astype(np.int64)
    n_int = n.copy()

    # ---- prepare mu/alpha (Aesara tensors) ----
    # mu = E[count] = pi * n
    mu_obs_all = pi * n_int + 1e-9
    alpha_all = delta

    # consider only rows with n > 0 for observed likelihood
    nonzero_idx = np.where(n_int > 0)[0]
    if nonzero_idx.size > 0:
        mu_obs    = at.take(mu_obs_all, nonzero_idx)
        if hasattr(alpha_all, "shape"):
            alpha_obs = at.take(alpha_all, nonzero_idx)
        else:
            alpha_obs = alpha_all

        nb_dist   = pm.NegativeBinomial.dist(mu=mu_obs, alpha=alpha_obs)
        logp_val  = pm.logp(nb_dist, obs_counts[nonzero_idx])
        pm.Potential(name=f"p_obs_{data_type}", var=at.sum(logp_val))
    else:
        # no informative rows; register a zero potential for completeness
        pm.Potential(name=f"p_obs_{data_type}", var=at.as_tensor_variable(0.0))

    # ---- predictive head (handle n==0 with large n_pred) ----
    zero_mask = (n_int == 0)
    if np.any(zero_mask):
        idx_list = obs_data.index.to_numpy()[zero_mask]
        warnings.warn(
            f"[neg_binom] effective_cases == 0 for {zero_mask.sum()} rows; "
            f"using n_pred={int(LARGE_N_FOR_PRED)} for predictive rate. indices={idx_list}",
            RuntimeWarning,
        )

    n_pred = n_int.copy()
    n_pred[zero_mask] = int(1e9)
    mu_pred = pi * n_pred + 1e-9
    
    # count_pred = pm.NegativeBinomial(name=f"p_count_{data_type}", mu=mu_pred, alpha=alpha_all)
    # pm.Deterministic(name=f"p_pred_{data_type}", var=count_pred / (n_pred + 1e-9))
    pm.Deterministic(name=f"p_pred_{data_type}", var=mu_pred / n_pred)

# beta_binom은 수동으로 구현한 것, beta_binom_2는 pymc 내장 함수로 구현한 것
# 내장함수로 구현한 걸 믿겠음. beta_binom_2 사용
def beta_binom(data_type, obs_data, pi, delta) -> None:
    # ---- read & validate ----
    p = obs_data['value'].to_numpy(dtype=float)           # prevalence in [0,1]
    n = obs_data['effective_sample_size'].to_numpy(dtype=int)

    assert np.all(p >= 0.0), "observed values must be non-negative"
    assert np.all(p <= 1.0), "observed values must be <= 1"
    assert np.all(n >= 0),   "effective sample size must be non-negative"

    # ---- observed counts (effective_cases) ----
    effective_cases = np.clip(np.rint(p * n), 0, n).astype(int)
    n_int = n.astype(int)

    # ---- mask rows with n==0 for observed likelihood ----
    mask = (n_int > 0)
    if not np.any(mask):
        # no informative rows; still register a zero potential for completeness
        pm.Potential(name=f"p_obs_{data_type}", var=at.as_tensor_variable(0.0))
    else:
        n_obs   = n_int[mask]
        k_obs   = effective_cases[mask]
        idx_obs = np.where(mask)[0]

        # concentration parameter (delta) may be scalar / vector / tensor
        if isinstance(delta, list) and len(delta) == 1:
            delta = delta[0]
        delta_t = at.as_tensor_variable(delta)

        # alpha,beta parameterization: alpha = pi * C, beta = (1-pi) * C
        pi_clip   = pm.math.clip(pi, EPS, 1.0 - EPS)
        alpha_all = pi_clip * delta_t
        beta_all  = (1.0 - pi_clip) * delta_t

        # take masked entries if vector/tensor, else keep scalar
        if hasattr(alpha_all, "shape") and (getattr(alpha_all, "ndim", 0) != 0):
            alpha_obs = at.take(alpha_all, idx_obs)
            beta_obs  = at.take(beta_all,  idx_obs)
        else:
            alpha_obs = alpha_all
            beta_obs  = beta_all

        pm.BetaBinomial(
            name=f"p_obs_{data_type}",
            n=n_obs,
            alpha=alpha_obs,
            beta=beta_obs,
            observed=k_obs,
        )

    # ---- predictive head: use large n for rows with n==0 ----
    zero_mask = (n_int == 0)
    if np.any(zero_mask):
        idx_list = obs_data.index.to_numpy()[zero_mask]
        warnings.warn(
            f"[beta_binom] effective_sample_size == 0 for {zero_mask.sum()} rows; "
            f"using n_pred={int(LARGE_N_FOR_PRED)} for predictive rate. indices={idx_list}",
            RuntimeWarning,
        )

    n_pred = n_int.astype(np.int64)
    n_pred[zero_mask] = int(LARGE_N_FOR_PRED)

    # reuse alpha,beta on full set (broadcast OK)
    if isinstance(delta, list) and len(delta) == 1:
        delta = delta[0]
    delta_t = at.as_tensor_variable(delta)
    pi_clip = pm.math.clip(pi, EPS, 1.0 - EPS)
    alpha_param = pi_clip * delta_t
    beta_param  = (1.0 - pi_clip) * delta_t

    count_pred = pm.BetaBinomial(
        name=f"p_count_{data_type}",
        n=n_pred,
        alpha=alpha_param,
        beta=beta_param,
    )
    pm.Deterministic(name=f"p_pred_{data_type}", var=count_pred / n_pred.astype(float))

def normal(data_type, obs_data, pi, sigma) -> None:
    p = obs_data['value'].to_numpy(dtype=float)
    s = obs_data['standard_error'].to_numpy(dtype=float)
    assert np.all(s >= 0.0), "standard error must be non-negative"

    # 분산/표준편차 (PyTensor)
    std = pm.math.sqrt(sigma**2 + at.as_tensor_variable(s)**2)

    # 관측 우도
    pm.Normal(
        name=f"p_obs_{data_type}",
        mu=pi,
        sigma=std,
        observed=p,
    )
    # predictive
    pm.Normal(
        name=f"p_pred_{data_type}",
        mu=pi,
        sigma=std,
    )

def log_normal(data_type, obs_data, pi, sigma) -> None:
    p = obs_data['value'].to_numpy(dtype=float)
    s = obs_data['standard_error'].to_numpy(dtype=float)

    # 디버그/경고
    bad_p = np.where(p <= 0)[0]
    bad_s = np.where(s <  0)[0]
    if bad_p.size:
        warnings.warn(f"[log_normal] non-positive p in {bad_p.size} rows; "
                      f"indices={obs_data.index.to_numpy()[bad_p]}", RuntimeWarning)
    if bad_s.size:
        warnings.warn(f"[log_normal] negative SE in {bad_s.size} rows; "
                      f"indices={obs_data.index.to_numpy()[bad_s]}", RuntimeWarning)

    # 유효성
    assert np.all(p > 0.0), "observed values must be positive"
    assert np.all(s >= 0.0), "standard error must be non-negative"

    # 관측 로그값 (NumPy)
    log_p_obs = np.log(np.clip(p, EPS, None))

    # Var[log p] ≈ (sigma)^2 + (s/p)^2
    s_over_p = at.as_tensor_variable(s / np.clip(p, EPS, np.inf))
    std = pm.math.sqrt(sigma**2 + s_over_p**2)

    # 모델 평균 (로그 스케일)
    log_pi = pm.math.log(pm.math.clip(pi, EPS, 1.0 - EPS))

    # 관측 우도
    pm.Normal(
        name=f"p_obs_{data_type}",
        mu=log_pi,
        sigma=std,
        observed=log_p_obs,
    )

    # 예측 (로그 → 원 스케일로 변환)
    log_p_pred = pm.Normal(name=f"p_log_pred_{data_type}", mu=log_pi, sigma=std)
    pm.Deterministic(name=f"p_pred_{data_type}", var=pm.math.exp(log_p_pred))

def offset_log_normal(data_type, obs_data, pi, sigma) -> None:
    # 읽기 + 체크
    p = obs_data['value'].to_numpy(dtype=float)
    s = obs_data['standard_error'].to_numpy(dtype=float)
    if np.any(p < 0.0):
        warnings.warn(f"[offset_log_normal] negative p in {np.sum(p<0)} rows; "
                      f"indices={obs_data.index.to_numpy()[p<0]}", RuntimeWarning)
    if np.any(s < 0.0):
        warnings.warn(f"[offset_log_normal] negative SE in {np.sum(s<0)} rows; "
                      f"indices={obs_data.index.to_numpy()[s<0]}", RuntimeWarning)
    assert np.all(p >= 0.0), "observed values must be non-negative"
    assert np.all(s >= 0.0), "standard error must be non-negative"

    # offset ζ ~ Uniform
    zeta = pm.Uniform(
        name=f"p_zeta_{data_type}",
        lower=EPS,
        upper=10.0,
        initval=1e-6,
    )

    # 텐서 변환
    p_t = at.as_tensor_variable(p)
    s_t = at.as_tensor_variable(s)

    # 로그 스케일의 관측/평균
    p_plus = p_t + zeta
    pi_plus = pm.math.clip(pi, 0.0, np.inf) + zeta

    # Var[log(p+ζ)] ≈ sigma^2 + (s/(p+ζ))^2
    std = pm.math.sqrt(sigma**2 + (s_t / pm.math.maximum(p_plus, EPS))**2)

    log_obs = pm.math.log(pm.math.maximum(p_plus, EPS))
    log_pi  = pm.math.log(pm.math.maximum(pi_plus, EPS))

    # 관측 우도 (Potential 방식)
    rv = pm.Normal.dist(mu=log_pi, sigma=std)
    logp_vec = pm.logp(rv, log_obs)
    pm.Potential(name=f"p_obs_likelihood_{data_type}", var=at.sum(logp_vec))

    # 예측 (로그 latent → 원 스케일, offset 제거)
    log_pred = pm.Normal(name=f"p_log_pred_{data_type}", mu=log_pi, sigma=std)
    pm.Deterministic(
        name=f"p_pred_{data_type}",
        var=pm.math.maximum(pm.math.exp(log_pred) - zeta, 0.0),  # 음수 방지
    )



# def neg_binom_lower_bound(data_type, obs_data, pi, delta) -> None:
#     """
#     Generate PyMC objects for a negative binomial lower bound model

#     Parameters
#     ----------
#     name  : str
#     pi    : Tensor or array, expected rate per unit sample size
#     delta : Tensor or array, dispersion (alpha) parameter
#     p     : array, observed rates
#     n     : array, sample sizes (to scale rates to counts)

#     Returns
#     -------
#     dict with keys:
#         - p_obs: potential log-likelihood enforcing lower bound
#     """

#     # --------------------------- 1) initialize pm_model ---------------------------   
#     pm_model = pm.modelcontext(None) # at reforged_mr/model/likelihood/neg_binom_lower_bound()


#     # --------------------------- 2) extract shared data ---------------------------   
#     data_type = pm_model.shared_data["data_type"]
#     data_type = f'lb_{data_type}'

#     lb_data = pm_model.shared_data["lb_data"]
#     p = lb_data['value'].to_numpy()
#     n = lb_data['effective_sample_size'].to_numpy().astype(int)


#     # 3) NumPy 형태로 변환 및 유효성 검사
#     p = np.asarray(p)
#     n = np.asarray(n, dtype=int)
#     assert np.all(p >= 0), 'observed values must be non-negative'
#     assert np.all(n > 0), 'effective sample size must be positive'


#     # Convert to integer counts
#     obs_counts = np.round(p * n).astype(int)
#     n_int = n.astype(int)

#     # Mean counts
#     mu = pi * n_int + 1e-9
#     # Lower-bound counts: max(obs, mu)
#     counts_lb = pm.math.maximum(obs_counts, mu)

#     # Negative binomial log-likelihood potential
#     dist = pm.NegativeBinomial.dist(mu=mu, alpha=delta)
#     # sum logp over observations
#     logp = dist.logp(counts_lb)
#     p_obs = pm.Potential(f'p_obs_{data_type}', pm.math.sum(logp))