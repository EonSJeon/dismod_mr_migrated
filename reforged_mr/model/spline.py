import numpy as np
import pymc as pm
import pytensor.tensor as at

# TODO: scipy.interpolate.interp1d에 comparable하게 여러 옵션 추가 
# i.e. quadratic, cubic, etc. 


# [Helper Function]: build weight matrix for linear interpolation
def build_weight_matrix_linear(knots: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """
    1차 선형 보간(linear interpolation)을 위한 weight 행렬 W를 사전 계산.
    """
    K = len(knots)
    N = len(ages)
    W = np.zeros((N, K), dtype=np.float64)

    idx = np.searchsorted(knots, ages, side="right")
    for i in range(N):
        a = ages[i]
        j_plus = idx[i]
        j_minus = j_plus - 1

        # (1) age == 마지막 knot인 경우 → W[i, K-1] = 1
        if j_plus == K and np.isclose(a, knots[-1]):
            W[i, K - 1] = 1.0
            continue

        # (2) knots 범위 바깥(왼쪽)에 있는 경우 → W[i,:] = 0
        if j_plus == 0:
            continue

        # (3) knots 범위 바깥(오른쪽)에 있는 경우 → W[i,:] = 0
        if j_minus == K - 1:
            continue

        # (4) 내부 구간: knots[j_minus] < a < knots[j_plus]
        left_knot = knots[j_minus]
        right_knot = knots[j_plus]
        if right_knot == left_knot:
            W[i, j_minus] = 1.0
        else:
            w_left = (right_knot - a) / (right_knot - left_knot)
            w_right = (a - left_knot) / (right_knot - left_knot)
            W[i, j_minus] = w_left
            W[i, j_plus] = w_right

    return W



# [Main Function]: define spline variables
def spline() -> at.TensorVariable:
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
    # --------------------------- 1) initialize pm_model ---------------------------   
    pm_model = pm.modelcontext(None) # at reforged_mr/model/spline/spline()


    # --------------------------- 2) extract shared data ---------------------------   
    data_type = pm_model.shared_data["data_type"]
    knots = pm_model.shared_data["knots"]
    K = len(knots)  # K: number of knots 
    ages = pm_model.shared_data["ages"]
    smoothing = pm_model.shared_data["smoothing"]
    interpolation_method = pm_model.shared_data["interpolation_method"]
    

    # --------------------------- 3) validate shared data ---------------------------   
    if interpolation_method != "linear":
        raise ValueError(
            "이 spline 구현은 'linear' 보간만 지원합니다. "
            f"interpolation_method={interpolation_method!r}"
        )
    if not np.all(np.diff(knots) > 0):
        raise ValueError("Spline knots must be strictly increasing.")


    # --------------------------- 4) calculate weight matrix ---------------------------   
    W_numpy = build_weight_matrix_linear(knots, ages)


    # --------------------------- 5) define spline variables ---------------------------   
    # 5-1) γ_k ~ Normal(0, 10^2) prior
    gamma = [
        pm.Normal(
            f"gamma_{data_type}_{i}",
            mu=0.0,
            sigma=10.0,  # σ = 10 → 분산 10^2
            initval=-10.0,
        )
        for i in range(K)
    ]

    print('printing type of gamma')
    print(type(gamma))
    print(gamma)

    gamma_vec = at.stack(gamma)  # shape=(K,)

    print('printing type of gamma_vec')
    print(type(gamma_vec))
    print(gamma_vec)

    exp_gamma = at.exp(gamma_vec)  # shape=(K,), 양수

    print('printing type of exp_gamma')
    print(type(exp_gamma))
    print(exp_gamma)

    # 5-2) W_numpy를 PyTensor 상수(Constant)로 변환
    W_t = at.constant(W_numpy)  # shape=(len(ages), K)

    print('printing type of W_t')
    print(type(W_t))
    print(W_t)

    # 5-3) mu_age 계산: W @ exp_gamma
    mu_age = at.dot(W_t, exp_gamma)  # shape=(len(ages),)

    print('printing type of mu_age before pm.Deterministic')
    print(type(mu_age))
    print(mu_age)

    pm.Deterministic(f"mu_age_{data_type}", mu_age)

    # 5-4) ||h'|| ~ Normal(0, σ^2) smoothing 페널티
    #     수식:
    #       γ_min = log( (∑_{i=0..K-1} exp(γ_i) / 10) / K )
    #       ||h'|| = sqrt( ∑_{k=0..K-2} [ max(γ_k,γ_min) - max(γ_{k+1},γ_min) ]^2
    #                        / [ (knots[k+1] - knots[k]) * (knots[K-1] - knots[0]) ] )


    # --------------------------- 6) define smoothing penalty ---------------------------   
    if (smoothing is not None) and (smoothing > 0) and np.isfinite(smoothing):
        # 1) γ_min 계산
        mean_term = at.sum(exp_gamma) / 10.0  # ∑ exp(γ_i) / 10
        gamma_min = at.log(mean_term / K)  # log( (∑ exp(γ_i)/10) / K )

        # 2) clipped_gamma_i = max(γ_i, γ_min)
        clipped_gamma = at.switch(
            gamma_vec < gamma_min, gamma_min, gamma_vec
        )  # shape=(K,)

        # 3) adjacent difference: clipped_gamma[k] - clipped_gamma[k+1]
        diffs = clipped_gamma[:-1] - clipped_gamma[1:]  # shape=(K-1,)

        # 4) denominator: (knots[k+1]-knots[k]) * (a_K - a_1)
        total_range = knots[-1] - knots[0]  # scalar
        intervals = knots[1:] - knots[:-1]  # shape=(K-1,)
        denom = intervals * total_range  # shape=(K-1,)
        denom_t = at.constant(denom)  # TensorConstant

        # 5) sum_term = ∑ [ diffs^2 / denom ]
        sum_term = at.sum((diffs**2) / denom_t)  # scalar

        # 6) 이제 로그우도에 바로 분산(σ^2) 사용
        #    -0.5 * (sum_term) / (σ^2)
        pm.Potential(f"smooth_{data_type}", -0.5 * sum_term / (smoothing**2))

        print('printing type of mu_age after pm.Deterministic')
        print(type(mu_age))
        print(mu_age)
        
        return mu_age