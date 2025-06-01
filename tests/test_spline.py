"""Test Spline Model (modern PyMC4 style) with full‐range plotting"""
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

import dismod_mr

def test_age_pattern_model_sim():
    # 1) simulate data
    np.random.seed(42)

    # ● 0부터 100까지 포함 (5 단위), 총 21개 → 관측용 knot 위치
    a = np.arange(0, 101, 5)         # [  0,  5, 10, ..., 100 ] (length=21)
    # 실제 연속함수 f_true(a) = 0.0001 * (a*(100−a) + 100)
    pi_true = 0.0001 * (a * (100. - a) + 100.)  # shape=(21,)
    sigma_true = 0.025 * np.ones_like(pi_true)  # shape=(21,)
    # noisy observations, truncated at zero
    p = np.random.normal(pi_true, sigma_true)
    p = np.maximum(0.0, p)

    # ● ages와 knots도 0~100을 포함하도록 (모델에 넣을 discrete age grid)
    ages  = np.arange(101)           # [0,1,2,...,100] (length=101)
    knots = np.arange(0, 101, 5)     # [0,5,10,...,100]  (length=21)

    # 2) build and fit the model
    with pm.Model():
        # spline priors + interpolant
        spline_vars = dismod_mr.model.spline.spline(
            'test',
            ages=ages,
            knots=knots,
            smoothing=0.005
        )
        mu_age = spline_vars['mu_age']  # deterministic age‐rates (shape=(101,))

        # pick out the pi at the discrete ages in `a` (knot 위치)
        pi = pm.Deterministic(
            "pi",
            mu_age[a]    # a는 [0,5,10,...,100] (length=21)
        )

        # likelihood  
        pm.Normal(
            "obs",
            mu=pi,
            sigma=sigma_true,
            observed=p
        )

        # sample
        trace = pm.sample(
            draws=4000,
            tune=500,
            chains=3,
            cores=1,
            return_inferencedata=False
        )

    # 3) posterior mean at the knot locations
    #    mu_age_test 은 shape=(n_samples, 101) 이므로 axis=0 평균
    post_mu = trace["mu_age_test"].mean(axis=0)  # shape (101,)

    # ──────────────────────────────────────────────────────────────────────
    # (A) knot 위치에서의 True vs Posterior 비교
    print("True vs. Posterior at knot ages:")
    for ki in knots:
        # pi_true[ki//5] 와 post_mu[ki] 를 비교
        print(f" age={ki:3d}  true={pi_true[ki//5]:.6f}  post={post_mu[ki]:.6f}")

    # (A) 플롯  — only at knots
    true_vals = pi_true                   # shape=(21,), 실제값 at knot
    post_vals = post_mu[knots.astype(int)]  # shape=(21,), posterior mean at knot

    plt.figure(figsize=(8, 4))
    plt.plot(knots, true_vals, 'o-', label='True π (at knots)', color='tab:blue')
    plt.plot(knots, post_vals, 'x--', label='Posterior mean (at knots)', color='tab:orange')
    plt.xlabel("Age")
    plt.ylabel("Rate")
    plt.title("True vs Posterior Mean at Knot Ages")
    plt.grid(alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # (B) 전체 연속 구간에서의 True 연속함수 vs Posterior spline 비교
    # 4-1) True 연속함수 f_true(a) 를 0~100 구간에서 세밀하게 계산
    a_grid = np.linspace(0, 100, 1001)   # 0,0.1,0.2,...,100
    f_true_grid = 0.0001 * (a_grid * (100. - a_grid) + 100.)  # continuous true

    # 4-2) posterior mean으로 산출된 mu_age 값도 ages=0..100 정수 지점마다 계산됨
    #      post_mu(shape=101,) 자체가 “각 정수 age에서 추정된 posterior mean”
    #      즉 post_mu[a] = mu_age posterior mean at age=a
    #    → 이 값을 선형 보간 없이 그대로 점선으로 연결

    plt.figure(figsize=(8, 4))
    plt.plot(a_grid, f_true_grid, '-', label='True continuous f(a)', color='tab:blue')
    plt.plot(ages, post_mu, 'x--', label='Posterior mean μ_age (discrete)', color='tab:orange')
    plt.xlabel("Age")
    plt.ylabel("Rate")
    plt.title("True continuous function vs Posterior mean μ_age over full age range")
    plt.grid(alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # 끝까지 정상 실행되면 테스트 성공
    print("✔️ Test and full‐range plots completed successfully.")

