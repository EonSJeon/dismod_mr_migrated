"""Test age group models."""
import numpy as np
import pymc as pm
import dismod_mr.testing.data_simulation
import matplotlib.pyplot as plt
import dismod_mr

def test_age_standardizing_approx():
    # 1) simulate data
    n = 50
    sigma_true = .025 * np.ones(n)

    # ages는 0부터 100까지 정수(101개)로 정의
    ages = np.arange(101)                               # [0,1,2,...,100]
    # 실제 true 함수도 ages에 맞춰 101개 점으로 계산
    pi_age_true = 0.0001 * (ages * (100. - ages) + 100.) # shape=(101,)

    # simulated_age_intervals 함수에는 “구간별 관측을 만들기 위한 나이 점(a)”이 필요합니다.
    # 원래 예제에서는 a = np.arange(0,100,1) (0부터 99까지 100개)를 썼습니다. 
    # 그대로 따라 하려면 아래와 같이 a를 다시 정의해 주세요.
    a = np.arange(0, 100, 1)  # [0,1,2,...,99], 길이 100

    # d에는 age_start, age_end, value 같은 컬럼이 들어 있습니다.
    d = dismod_mr.testing.data_simulation.simulated_age_intervals(
        'p',
        n,
        a,
        pi_age_true[a],  # pi_age_true는 길이 101이므로, a(0~99)를 인덱싱
        sigma_true
    )

    # 2) 모델 정의 및 샘플링
    with pm.Model():
        # 2-1) spline model: mu_age라는 심볼릭 텐서를 생성
        variables = {}
        variables.update(
            dismod_mr.model.spline.spline(
                'test',
                ages,                     # [0..100]
                knots=np.arange(0, 100, 5),  # [0,5,10,...,95]
                smoothing=0.005
            )
        )
        # variables['mu_age']는 shape=(101,)인 심볼릭 텐서

        # 2-2) age-standardize approximation
        age_weights = np.ones_like(ages)  # 길이 101, 모두 1
        variables.update(
            dismod_mr.model.age_groups.age_standardize_approx(
                'test',
                age_weights,
                variables['mu_age'],
                d['age_start'],   # simulated_age_intervals에서 생성된 NumPy array
                d['age_end'],     # shape = (n,) 
                ages              # shape = (101,)
            )
        )
        # variables['mu_interval']는 shape=(n,)인 심볼릭 텐서

        variables['pi'] = variables['mu_interval']

        # 2-3) likelihood
        dismod_mr.model.likelihood.normal(
            'test',
            pi=variables['pi'],
            sigma=0,
            p=d['value'],
            s=sigma_true
        )

        # 2-4) NUTS sampler로 아주 적은 샘플링 (예제용)
        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=3,
            cores=4,
            return_inferencedata=False
        )

    # 3) posterior mean of mu_age
    mu_age_samples = trace["mu_age_test"]  # shape = (3, 101)
    mu_age_mean = mu_age_samples.mean(axis=0)  # shape = (101,)

    # 4) plot true vs posterior
    plt.figure(figsize=(8, 4))
    plt.plot(ages, pi_age_true,    label="True π_age",    linewidth=2)
    plt.plot(ages, mu_age_mean,    label="Posterior μ_age", linestyle="--")
    plt.xlabel("Age")
    plt.ylabel("Rate")
    plt.title("True vs Posterior Age‐specific Rates")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5) diagnostics print
    print("Posterior μ_age at ages [0,50,100]:", mu_age_mean[[0,50,100]])




def test_age_integrating_midpoint_approx():
    # simulate data
    n = 50
    sigma_true = .025 * np.ones(n)
    a = np.arange(0, 100, 1)
    pi_age_true = .0001 * (a * (100. - a) + 100.)
    ages = np.arange(101)
    d = dismod_mr.testing.data_simulation.simulated_age_intervals(
        'p', n, a, pi_age_true, sigma_true
    )

    with pm.Model():
        # spline model
        variables = {}
        variables.update(
            dismod_mr.model.spline.spline(
                'test',
                ages,
                knots=np.arange(0, 101, 5),
                smoothing=.01
            )
        )
        # midpoint integration approximation
        variables.update(
            dismod_mr.model.age_groups.midpoint_approx(
                'test',
                variables['mu_age'],
                d['age_start'],
                d['age_end'],
                ages
            )
        )
        variables['pi'] = variables['mu_interval']
        # likelihood
        dismod_mr.model.likelihood.normal(
            'test',
            pi=variables['pi'],
            sigma=1e-6,  # small non-zero sigma
            p=d['value'],
            s=sigma_true
        )

        # use Adaptive Metropolis only
        step = pm.Metropolis()
        trace = pm.sample(
            draws=3,
            tune=0,
            step=step,
            chains=1,
            cores=1,
            progressbar=False,
            return_inferencedata=False
        )
        print(trace)
