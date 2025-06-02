import numpy as np
import pymc as pm
import pytensor.tensor as at

def build_weight_matrix_linear(knots: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """
    1차 선형 보간(linear interpolation)을 위한 weight 행렬 W를 사전 계산.
    """
    knots = np.asarray(knots, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)

    K = len(knots)
    N = len(ages)
    W = np.zeros((N, K), dtype=np.float64)

    idx = np.searchsorted(knots, ages, side='right')
    for i in range(N):
        a = ages[i]
        j_plus  = idx[i]
        j_minus = j_plus - 1

        # (1) age == 마지막 knot인 경우 → W[i, K-1] = 1
        if j_plus == K and a == knots[-1]:
            W[i, K-1] = 1.0
            continue

        # (2) knots 범위 바깥(왼쪽)에 있는 경우 → W[i,:] = 0
        if j_plus == 0:
            continue

        # (3) knots 범위 바깥(오른쪽)에 있는 경우 → W[i,:] = 0
        if j_minus == K-1:
            continue

        # (4) 내부 구간: knots[j_minus] < a < knots[j_plus]
        left_knot  = knots[j_minus]
        right_knot = knots[j_plus]
        if right_knot == left_knot:
            W[i, j_minus] = 1.0
        else:
            w_left  = (right_knot - a) / (right_knot - left_knot)
            w_right = (a - left_knot)   / (right_knot - left_knot)
            W[i, j_minus] = w_left
            W[i, j_plus]  = w_right

    return W


def spline(
    data_type: str,
    ages: np.ndarray,
    knots: np.ndarray,
    smoothing: float,                  # σ (||h'|| ~ Normal(0, σ^2))
    interpolation_method: str = 'linear',
) -> dict:
    """
    첨부된 수식
      h(a) = sum_{k=1..K-1} 1[a_k <= a < a_{k+1}] (
                (a - a_k)/(a_{k+1}-a_k) * e^{γ_k} +
                (a_{k+1} - a)/(a_{k+1}-a_k) * e^{γ_{k+1}}
              )
    γ_k ~ Normal(0, 10^2),
    ||h'|| ~ Normal(0, σ^2)

    을 그대로 반영한 spline 함수입니다.
    """
    assert pm.modelcontext(None) is not None, 'spline must be called within a PyMC model'
    # --- 1) 입력 검증 및 배열 변환 ---
    ages  = np.asarray(ages, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)

    if interpolation_method != 'linear':
        raise ValueError(
            "이 spline 구현은 'linear' 보간만 지원합니다. "
            f"interpolation_method={interpolation_method!r}"
        )
    if not np.all(np.diff(knots) > 0):
        raise ValueError("Spline knots must be strictly increasing.")


    # --- 3) W 행렬을 미리 계산 (NumPy) ---
    W_numpy = build_weight_matrix_linear(knots, ages)

    # --- 4) PyMC 모델 문맥 안에서 spline 변수 정의 ---

    K = len(knots)  # knot 개수

    # ────────────────────────────────────────────────────────────────
    # 4-1) γ_k ~ Normal(0, 10^2) prior
    gamma = [
        pm.Normal(
            f"gamma_{data_type}_{i}",
            mu=0.0,
            sigma=10.0,    # σ = 10 → 분산 10^2
            initval=0.0
        )
        for i in range(K)
    ]
    gamma_vec = at.stack(gamma)        # shape=(K,)
    exp_gamma = at.exp(gamma_vec)      # shape=(K,), 양수

    # ────────────────────────────────────────────────────────────────
    # 4-2) W_numpy를 PyTensor 상수(Constant)로 변환
    W_t = at.constant(W_numpy)          # shape=(len(ages), K)

    # 4-3) mu_age 계산: W @ exp_gamma
    mu_age = at.dot(W_t, exp_gamma)     # shape=(len(ages),)
    pm.Deterministic(f"mu_age_{data_type}", mu_age)

    # ────────────────────────────────────────────────────────────────
    # 4-4) ||h'|| ~ Normal(0, σ^2) smoothing 페널티
    #     수식:
    #       γ_min = log( (∑_{i=0..K-1} exp(γ_i) / 10) / K )
    #       ||h'|| = sqrt( ∑_{k=0..K-2} [ max(γ_k,γ_min) - max(γ_{k+1},γ_min) ]^2 
    #                        / [ (knots[k+1] - knots[k]) * (knots[K-1] - knots[0]) ] )
    #
    if (smoothing is not None) and (smoothing > 0) and np.isfinite(smoothing):
        # 1) γ_min 계산
        mean_term = at.sum(exp_gamma) / 10.0      # ∑ exp(γ_i) / 10
        gamma_min = at.log(mean_term / K)         # log( (∑ exp(γ_i)/10) / K )

        # 2) clipped_gamma_i = max(γ_i, γ_min)
        clipped_gamma = at.switch(gamma_vec < gamma_min,
                                    gamma_min,
                                    gamma_vec)    # shape=(K,)

        # 3) adjacent difference: clipped_gamma[k] - clipped_gamma[k+1]
        diffs = clipped_gamma[:-1] - clipped_gamma[1:]  # shape=(K-1,)

        # 4) denominator: (knots[k+1]-knots[k]) * (a_K - a_1)
        total_range = knots[-1] - knots[0]               # scalar
        intervals = knots[1:] - knots[:-1]               # shape=(K-1,)
        denom = intervals * total_range                  # shape=(K-1,)
        denom_t = at.constant(denom)                     # TensorConstant

        # 5) sum_term = ∑ [ diffs^2 / denom ]
        sum_term = at.sum((diffs ** 2) / denom_t)         # scalar

        # 6) 이제 로그우도에 바로 분산(σ^2) 사용
        #    -0.5 * (sum_term) / (σ^2)
        pm.Potential(
            f"smooth_{data_type}",
            -0.5 * sum_term / (smoothing ** 2)
        )

    # --- 5) 결과 반환 ---
    return {
        'gamma':  gamma,
        'mu_age': mu_age,
        'ages':   ages,
        'knots':  knots
    }
