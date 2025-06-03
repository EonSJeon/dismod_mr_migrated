import numpy as np
import pymc as pm
import pytensor.tensor as at


def binom(name, pi, p, n):
    """
    Generate PyMC objects for a binomial model

    Parameters
    ----------
    name : str
    pi   : Tensor or array, success probability for each observation
    p    : array, observed proportions
    n    : array, effective sample sizes

    Returns
    -------
    dict
        - p_obs: observed Binomial RV
        - p_pred: posterior predictive proportion (Deterministic)
    """
    assert pm.modelcontext(None) is not None, 'Must be called within a PyMC model'
    # input validation
    assert np.all(p >= 0), 'observed values must be non-negative'
    assert np.all(n >= 0), 'effective sample size must non-negative'

    # observed counts (rounded)
    obs_counts = np.round(p * n).astype(int)
    n_int = n.astype(int)

    # observed likelihood
    p_obs = pm.Binomial(
        name=f'p_obs_{name}',
        n=n_int,
        p=pi + 1e-9,
        observed=obs_counts
    )

    # adjust zero-sample cases for predictive validity
    n_nonzero = n_int.copy()
    n_nonzero[n_nonzero == 0] = int(1e6)

    # latent predictive counts
    p_count = pm.Binomial(
        name=f'p_count_{name}',
        n=n_nonzero,
        p=pi + 1e-9
    )
    # convert to proportions
    p_pred = pm.Deterministic(
        name=f'p_pred_{name}',
        var=p_count / n_nonzero
    )

    return {'p_obs': p_obs, 'p_pred': p_pred}

def poisson(name, pi, p, n):
    """
    Generate PyMC objects for a Poisson model

    Parameters
    ----------
    name : str
    pi   : Tensor or array, expected rate per unit sample size
    p    : array, observed rates
    n    : array, sample sizes (to scale rates to counts)

    Returns
    -------
    dict with keys:
      - p_obs: observed Poisson RV
      - p_pred: posterior predictive rate
    """
    assert pm.modelcontext(None) is not None, 'poisson() must be called within a PyMC model'
    assert np.all(p >= 0), 'observed values must be non-negative'
    assert np.all(n >= 0), 'effective sample size must be non-negative'

    obs_counts = np.round(p * n).astype(int)
    n_float = n.astype(float)

    p_obs = pm.Poisson(
        name=f'p_obs_{name}',
        mu=pi * n_float,
        observed=obs_counts
    )

    n_pred = n_float.copy()
    n_pred[n_pred == 0] = 1e6
    count_pred = pm.Poisson(
        name=f'p_count_{name}',
        mu=pi * n_pred
    )
    p_pred = pm.Deterministic(
        name=f'p_pred_{name}',
        var=count_pred / n_pred
    )

    return {'p_obs': p_obs, 'p_pred': p_pred}





def neg_binom(
    name: str,
    pi,               # TensorVariable 또는 array‐like
    delta,            # TensorVariable 또는 스칼라
    p: np.ndarray,    # 관측된 비율 (np.ndarray)
    n: np.ndarray     # 표본 크기 (np.ndarray of ints)
) -> dict:
    """
    PyMC 5.3용 negative‐binomial likelihood 함수

    Returns
    -------
    dict with keys:
      - p_obs : None (로그우도는 pm.Potential으로 걸림)
      - p_pred: pm.Deterministic (posterior predictive 비율, shape=(N,))
    """

    # 1) 반드시 pm.Model() 내부에서 호출되어야 함
    assert pm.modelcontext(None) is not None, 'neg_binom() must be called within a PyMC model'


    # 2) delta가 1‐원소 리스트로 넘어왔을 때, 첫번째 원소를 사용
    if isinstance(delta, list) and len(delta) == 1:
        delta = delta[0]

    # 3) NumPy 형태로 변환 및 유효성 검사
    p = np.asarray(p)
    n = np.asarray(n, dtype=int)
    assert np.all(p >= 0), "observed values must be non‐negative"
    assert np.all(n >= 0), "effective sample size must be non‐negative"

    # 4) “비율 * 표본크기” → 정수 카운트로 변환
    obs_counts = np.round(p * n).astype(int)  
    n_int      = n.copy()                    

    # 5) 표본 크기가 0보다 큰 인덱스를 미리 구한다
    nonzero_idx = np.where(n_int > 0)[0]     

    # 6) TensorVariable 형태로 “mu_obs_all” 계산
    mu_obs_all = pi * n_int + 1e-9            
    alpha_all  = delta                        

    # 7) “표본크기>0” 부분만 뽑아낸다
    mu_obs = at.take(mu_obs_all, nonzero_idx)           
    if hasattr(alpha_all, "shape"):
        alpha_obs = at.take(alpha_all, nonzero_idx)
    else:
        alpha_obs = alpha_all

    # 8) 관측 likelihood → Potential
    nb_dist   = pm.NegativeBinomial.dist(mu=mu_obs, alpha=alpha_obs)
    logp_val  = pm.logp(nb_dist, obs_counts[nonzero_idx])  
    summed_lp = at.sum(logp_val)
    pm.Potential(name=f"p_obs_{name}", var=summed_lp)

    # 9) posterior predictive: n_pred에서 0인 표본만 큰 값으로 치환
    n_pred = n_int.copy()
    n_pred[n_pred == 0] = int(1e9)         
    mu_pred = pi * n_pred + 1e-9            

    # 10) 예측된 비율을 Deterministic으로 등록
    p_pred = pm.Deterministic(
        name=f"p_pred_{name}",
        var=mu_pred / n_pred                
    )

    return {"p_obs": None, "p_pred": p_pred}







def neg_binom_lower_bound(name, pi, delta, p, n):
    """
    Generate PyMC objects for a negative binomial lower bound model

    Parameters
    ----------
    name  : str
    pi    : Tensor or array, expected rate per unit sample size
    delta : Tensor or array, dispersion (alpha) parameter
    p     : array, observed rates
    n     : array, sample sizes (to scale rates to counts)

    Returns
    -------
    dict with keys:
      - p_obs: potential log-likelihood enforcing lower bound
    """
    assert pm.modelcontext(None) is not None, 'neg_binom_lower_bound() must be called within a PyMC model'
    assert np.all(p >= 0), 'observed values must be non-negative'
    assert np.all(n > 0), 'effective sample size must be positive'

    # Convert to integer counts
    obs_counts = np.round(p * n).astype(int)
    n_int = n.astype(int)

    # Mean counts
    mu = pi * n_int + 1e-9
    # Lower-bound counts: max(obs, mu)
    counts_lb = pm.math.maximum(obs_counts, mu)

    # Negative binomial log-likelihood potential
    dist = pm.NegativeBinomial.dist(mu=mu, alpha=delta)
    # sum logp over observations
    logp = dist.logp(counts_lb)
    p_obs = pm.Potential(f'p_obs_{name}', pm.math.sum(logp))

    return {'p_obs': p_obs}


# beta_binom은 수동으로 구현한 것, beta_binom_2는 pymc 내장 함수로 구현한 것
# 내장함수로 구현한 걸 믿겠음. beta_binom_2 사용
def beta_binom(name, pi, delta, p, n):
    """
    Generate PyMC objects for a beta-binomial model with faster computation

    Parameters
    ----------
    name : str
    pi   : Tensor or array, expected success probabilities
    delta: Tensor or array, dispersion (pseudocount fraction) parameter
    p    : array, observed proportions
    n    : array, effective sample sizes

    Returns
    -------
    dict with keys:
      - p_obs: observed BetaBinomial RV
      - p_pred: posterior predictive proportion
    """
    assert np.all(p >= 0), 'observed values must be non-negative'
    assert np.all(n >= 0), 'effective sample size must be non-negative'
    assert pm.modelcontext(None) is not None, 'beta_binom() must be called within a PyMC model'

    # Observed counts and integer sizes
    obs_counts = np.round(p * n).astype(int)
    n_int = n.astype(int)

    mask = n_int > 0
    # Parameterize BetaBinomial with alpha, beta
    alpha_param = pi * delta * 50
    beta_param = (1 - pi) * delta * 50

    p_obs = pm.BetaBinomial(
        name=f'p_obs_{name}',
        n=n_int[mask],
        alpha=alpha_param[mask] if hasattr(alpha_param, 'shape') else alpha_param,
        beta=beta_param[mask] if hasattr(beta_param, 'shape') else beta_param,
        observed=obs_counts[mask]
    )

    # Posterior predictive counts: replace zero-sample cases
    n_pred = n_int.copy()
    n_pred[n_pred == 0] = int(1e9)

    count_pred = pm.BetaBinomial(
        name=f'p_count_{name}',
        n=n_pred,
        alpha=alpha_param,
        beta=beta_param
    )
    p_pred = pm.Deterministic(
        name=f'p_pred_{name}',
        var=count_pred / n_pred
    )

    return {'p_obs': p_obs, 'p_pred': p_pred}



def normal(name, pi, sigma, p, s):
    """
    Generate PyMC objects for a normal model

    Parameters
    ----------
    name  : str
    pi    : Tensor or array, expected values
    sigma : Tensor or array, model dispersion parameter
    p     : array, observed values
    s     : array, observational standard errors

    Returns
    -------
    dict with keys:
      - p_obs: observed Normal RV
      - p_pred: posterior predictive Normal RV
    """
    assert pm.modelcontext(None) is not None, 'normal() must be called within a PyMC model'
    p = np.array(p)
    s = np.array(s)
    assert np.all(s >= 0), 'standard error must be non-negative'

    var = sigma**2 + s**2
    std = np.sqrt(var)

    p_obs = pm.Normal(
        name=f'p_obs_{name}',
        mu=pi,
        sigma=std,
        observed=p
    )
    p_pred = pm.Normal(
        name=f'p_pred_{name}',
        mu=pi,
        sigma=std
    )
    return {'p_obs': p_obs, 'p_pred': p_pred}





def log_normal(data_type, pi, sigma, p, s):
    """
    Generate PyMC objects for a lognormal model on log scale with observational error.
    수정된 부분: Normal과 Deterministic 이름 충돌 해결
    """
    assert pm.modelcontext(None) is not None, 'log_normal() must be called within a PyMC model'

    # 1) NumPy array로 변환
    p = np.array(p)
    s = np.array(s)

    # 디버그: 잘못된 인덱스 찾아 출력
    bad_p_idx = np.where(p <= 0)[0]
    bad_s_idx = np.where(s < 0)[0]
    if bad_p_idx.size > 0:
        print(f"[DEBUG] data_type={data_type}: p 배열에서 0 이하인 값 (총 {bad_p_idx.size}개):")
        for i in bad_p_idx:
            print(f"    index={i}, p[{i}]={p[i]}")
    if bad_s_idx.size > 0:
        print(f"[DEBUG] data_type={data_type}: s 배열에서 음수인 값 (총 {bad_s_idx.size}개):")
        for i in bad_s_idx:
            print(f"    index={i}, s[{i}]={s[i]}")

    # 2) 관측값 유효성 검사
    assert np.all(p > 0), 'observed values must be positive'
    assert np.all(s >= 0), 'standard error must be non-negative'

    # 3) 관측 로그값은 NumPy로 미리 계산
    log_p_np = np.log(p)

    # 4) 관측 분산 및 표준편차 (PyTensor)
    var = sigma**2 + (s / p)**2
    std = pm.math.sqrt(var)

    # 5) 모델링
    log_pi = pm.math.log(pi + 1e-9)

    p_obs = pm.Normal(
        name=f'p_obs_{data_type}',
        mu=log_pi,
        sigma=std,
        observed=log_p_np
    )

    # 1) 로그 예측치 RV 이름: p_log_pred_...
    log_p_pred = pm.Normal(
        name=f'p_log_pred_{data_type}',  
        mu=log_pi,
        sigma=std
    )

    # 2) 역변환된 예측치 Deterministic 이름은 p_pred_... 로 다르게 설정
    p_pred = pm.Deterministic(
        name=f'p_pred_{data_type}',      
        var=pm.math.exp(log_p_pred)
    )

    return {'p_obs': p_obs, 'p_pred': p_pred}




def offset_log_normal(name, pi, sigma, p, s):
    """
    Generate PyMC objects for an offset log-normal model

    Parameters
    ----------
    name  : str
    pi    : Tensor or array, expected baseline values
    sigma : Tensor or array, model dispersion parameter on log scale
    p     : array, observed values of rates
    s     : array, observational standard errors

    Returns
    -------
    dict with keys:
      - p_zeta: Uniform prior for offset
      - p_obs: observed Normal RV on log(p + zeta)
      - p_pred: posterior predictive back-transformed minus offset
    """
    assert pm.modelcontext(None) is not None, 'offset_log_normal() must be called within a PyMC model'
    p = np.array(p)
    s = np.array(s)
    assert np.all(p >= 0), 'observed values must be non-negative'
    assert np.all(s >= 0), 'standard error must be non-negative'

    # prior on offset to avoid log(0)
    p_zeta = pm.Uniform(
        name=f'p_zeta_{name}',
        lower=1e-9,
        upper=10.0,
        initval=1e-6
    )

    # observational variance on log scale: sigma^2 + (s/(p+zeta))^2
    # mask infinite s
    mask = ~np.isinf(s)
    var = sigma**2 + np.where(mask, (s/(p + p_zeta))**2, 0.0)
    std = np.sqrt(var)

    # observed log-transformed data
    log_obs = pm.math.log(p + p_zeta)
    log_pi = pm.math.log(pi + p_zeta)

    p_obs = pm.Normal(
        name=f'p_obs_{name}',
        mu=log_pi[mask],
        sigma=std[mask],
        observed=log_obs[mask]
    )

    # predictive log-ratio
    log_pred = pm.Normal(
        name=f'log_pred_{name}',
        mu=log_pi,
        sigma=std
    )

    # back-transform and remove offset
    p_pred = pm.Deterministic(
        name=f'p_pred_{name}',
        var=(pm.math.exp(log_pred) - p_zeta)
    )

    return {'p_zeta': p_zeta, 'p_obs': p_obs, 'p_pred': p_pred}




# def beta_binom(name, pi, p, n):
#     """
#     Generate PyMC objects for a beta-binomial model

#     Parameters
#     ----------
#     name : str
#     pi   : Tensor or array, expected success probabilities
#     p    : array, observed proportions
#     n    : array, effective sample sizes

#     Returns
#     -------
#     dict with keys:
#       - p_n: Uniform prior on precision (pseudocount)
#       - pi_latent: array of Beta RVs per observation
#       - p_obs: observed Binomial RV with latent p
#       - p_pred: posterior predictive proportions
#     """
#     assert np.all(p >= 0), 'observed values must be non-negative'
#     assert np.all(n >= 0), 'effective sample size must be non-negative'

#     # Prior on precision (pseudocount)
#     p_n = pm.Uniform(
#         name=f'p_n_{name}',
#         lower=1e4,
#         upper=1e9,
#         initval=1e4
#     )

#     # Latent success probabilities per observation
#     # shape=len(pi)
#     pi_latent = pm.Beta(
#         name=f'pi_latent_{name}',
#         alpha=pi * p_n,
#         beta=(1 - pi) * p_n,
#         shape=pi.shape
#     )

#     # Observed counts and sample sizes
#     obs_counts = np.round(p * n).astype(int)
#     n_int = n.astype(int)

#     # Only include nonzero n for likelihood
#     mask = n_int > 0
#     p_obs = pm.Binomial(
#         name=f'p_obs_{name}',
#         n=n_int[mask],
#         p=pi_latent[mask],
#         observed=obs_counts[mask]
#     )

#     # Predictive counts: replace zero-sample with large N
#     n_pred = n_int.copy()
#     n_pred[n_pred == 0] = int(1e6)
#     p_count = pm.Binomial(
#         name=f'p_count_{name}',
#         n=n_pred,
#         p=pi_latent
#     )
#     p_pred = pm.Deterministic(
#         name=f'p_pred_{name}',
#         var=p_count / n_pred
#     )

#     return {'p_n': p_n, 'pi_latent': pi_latent, 'p_obs': p_obs, 'p_pred': p_pred}