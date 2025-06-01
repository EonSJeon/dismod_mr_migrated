"""Test covariate and process models."""
import numpy as np
import pandas as pd
import pymc as pm

import dismod_mr

def test_covariate_model_sim_no_hierarchy():
    # 1) Simulate normal data
    np.random.seed(42)
    model_data = dismod_mr.data.ModelData()
    model_data.hierarchy, model_data.output_template = dismod_mr.testing.data_simulation.small_output()

    # X ~ Normal(0,1)
    X = np.random.normal(0.0, 1.0, size=(128, 3))
    beta_true = np.array([-0.1, 0.1, 0.2])
    Y_true = X @ beta_true

    pi_true = np.exp(Y_true)
    sigma_true = 0.01 * np.ones_like(pi_true)
    # p ~ Normal(pi_true, sigma_true)
    p = np.random.normal(pi_true, sigma_true)

    # 2) Prepare input DataFrame
    df = pd.DataFrame({
        "value":      p,
        "x_0":        X[:, 0],
        "x_1":        X[:, 1],
        "x_2":        X[:, 2],
        "area":    ["all"] * len(p),
        "sex":     ["total"] * len(p),
        "year_start": [2000] * len(p),
        "year_end":   [2000] * len(p),
    })
    model_data.input_data = df

    # 3) Build and sample the PyMC model
    with pm.Model():
        # mean_covariate_model will register all RVs (alpha, beta, etc.) into this model
        variables = dismod_mr.model.covariates.mean_covariate_model(
            data_type="test",
            mu=1.0,
            input_data=model_data.input_data,
            parameters={},       # no custom priors
            model=model_data,
            root_area="all",
            root_sex="total",
            root_year="all",
            zero_re=True,
        )

        # Likelihood: p ~ Normal(pi, sigma_true)
        pm.Normal(
            "obs",
            mu=variables["pi"],
            sigma=sigma_true,
            observed=p
        )

        # Sample with NUTS (or Metropolis if you prefer)
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=2,
            target_accept=0.9,
            return_inferencedata=False
        )

    # 4) Print posterior means of each β
    print("Posterior means for β:")
    for i in range(3):
        name = f"beta_test_x_{i}"
        print(f"  {name} ≃ {trace[name].mean():.3f}")

def test_covariate_model_sim_w_hierarchy():
    np.random.seed(123)
    n = 10000

    # -----------------------------------
    # (0) hierarchy, output_template 가져오기
    # -----------------------------------
    hierarchy, output_template = dismod_mr.testing.data_simulation.small_output()

    # -----------------------------------
    # (1) 무작위로 area, sex, year 생성
    #    - area 는 'all', 'USA', 'CAN' 중 하나
    #    - sex 은 'male','female','total' 중 하나
    #    - year 은 1990~2010 사이 정수 랜덤
    # -----------------------------------
    area = np.random.choice(['all','USA','CAN'], size=n, p=[0.3, 0.3, 0.4])
    sex  = np.random.choice(['male','female','total'], size=n, p=[0.3, 0.3, 0.4])
    year = np.random.randint(1990, 2011, size=n)

    # -----------------------------------
    # (2) “계층상 모든 노드”에 대한 true α 값 정의
    #    여기서 U.columns 에 보니 ['super-region-1','NAHI','CAN','USA'] 네 개입니다.
    #    ‘all’(root)은 fixed intercept으로 처리되어 U.columns에 없음 → α=0.0
    #
    #    super-region-1 (α=0.0), NAHI (α=0.0), USA (α=0.1), CAN (α=-0.2)
    # -----------------------------------
    alpha_true = {
        'super-region-1': 0.0,
        'NAHI':           0.0,
        'USA':            0.1,
        'CAN':           -0.2
    }

    # -----------------------------------
    # (3) 각 관측치마다 “rand_term = U·α” 계산 (log π = rand_term + fix_term)
    #     모델에서는 β(고정효과)를 0으로 두고 mu=1 로 설정할 예정임 → fix_term=0
    #     따라서 log π_true = sum(α상위노드) 이고, π_true = exp(log π_true)
    # -----------------------------------
    log_pi_true = np.zeros(n, dtype=float)
    for i, a in enumerate(area):
        if a == 'all':
            # root area: αs 모두 0 → log π_true = 0 → π_true = 1
            log_pi_true[i] = 0.0
        elif a == 'USA':
            # USA 경로: super-region-1 → NAHI → USA
            log_pi_true[i] = (
                alpha_true['super-region-1']
                + alpha_true['NAHI']
                + alpha_true['USA']
            )
        elif a == 'CAN':
            # CAN 경로: super-region-1 → NAHI → CAN
            log_pi_true[i] = (
                alpha_true['super-region-1']
                + alpha_true['NAHI']
                + alpha_true['CAN']
            )
        else:
            raise ValueError(f"알 수 없는 area: {a}")

    # π_true 스케일로 변환
    pi_true = np.exp(log_pi_true)

    # -----------------------------------
    # (4) 시뮬레이션 노이즈 (π 스케일에서 σ=0.05 고정)
    # -----------------------------------
    sigma_true = 0.05 * np.ones_like(pi_true)

    # -----------------------------------
    # (5) 관측치 생성: p_obs ~ Normal(pi_true, sigma_true)
    #     → 모델에서는 “obs ∼ Normal(pi, sigma_true)” 로 맞춰 줘야 함
    # -----------------------------------
    p_obs = np.random.normal(loc=pi_true, scale=sigma_true)

    # -----------------------------------
    # (6) ModelData에 “p_obs”를 넣어 줌
    # -----------------------------------
    model_data = dismod_mr.data.ModelData()
    model_data.input_data = pd.DataFrame({
        'value':      p_obs,   # 관측된 π (실수)
        'area':       area,
        'sex':        sex,
        'year_start': year,
        'year_end':   year
    })
    model_data.hierarchy       = hierarchy
    model_data.output_template = output_template

    # -----------------------------------
    # (7) PyMC5 모델 실행
    #     → mean_covariate_model 내부에서 π = mu * exp(U·α + X·β) 구조를 사용
    #     → 여기서는 β는 고정효과가 없으므로 모두 0으로 남게 하고, mu=1 고정
    # -----------------------------------
    with pm.Model():
        variables = dismod_mr.model.covariates.mean_covariate_model(
            data_type='test',
            mu=1,                       # intercept(μ)=1 고정
            input_data=model_data.input_data,
            parameters={},              # 고정효과/랜덤효과 사전 옵션 없음 → 기본 Normal prior 사용
            model=model_data,
            root_area='all',
            root_sex='total',
            root_year='all',
            zero_re=False
        )

        # variables['pi'] 는 “π = mu * exp(U·α + X·β)” 형태(실수 스케일)로 반환
        pm.Normal(
            'obs',
            mu=variables['pi'],    # π 예측치
            sigma=sigma_true,      # 시뮬레이션 시 쓴 σ
            observed=p_obs         # 관측된 π 값
        )

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=4,
            target_accept=0.9,
            return_inferencedata=True
        )

    # -----------------------------------
    # (8) Posterior α 평균 출력
    # -----------------------------------
    print("=== Random effects (alpha) posterior mean ===")
    for node in variables["U"].columns:
        varname = f"alpha_test_{node}"
        arr = idata.posterior[varname].values  # shape=(chain, draw)
        print(f"{varname} ≃ {arr.mean():.3f}")

    # -----------------------------------
    # (9) Posterior β 평균 출력
    # -----------------------------------
    print("\n=== Fixed effects (beta) posterior mean ===")
    for effect in variables["X"].columns:
        varname = f"beta_test_{effect}"
        arr = idata.posterior[varname].values
        print(f"{varname} ≃ {arr.mean():.3f}")

    # -----------------------------------
    # (10) 간단한 Assert
    # -----------------------------------
    assert 'sex' not in variables['U']
    assert 'x_sex' in variables['X']
    assert len(variables['beta']) == 1

def test_fixed_effect_priors():
    model_data = dismod_mr.data.ModelData()
    model_data.hierarchy, model_data.output_template = (
        dismod_mr.testing.data_simulation.small_output()
    )
    params = {
        "fixed_effects": {
            "x_sex": {
                "dist":   "TruncatedNormal",
                "mu":      1.0,
                "sigma":   0.5,
                "lower":  -10.0,
                "upper":  10.0,
            }
        }
    }

    n = 32
    sex = np.random.choice(["male", "female", "total"], size=n, p=[0.3, 0.3, 0.4])
    beta_true = {"male": -1.0, "total": 0.0, "female": 1.0}
    pi_true = np.exp([beta_true[s] for s in sex])
    sigma_true = 0.05
    p = np.random.normal(loc=pi_true, scale=sigma_true)

    df = pd.DataFrame({"value": p, "sex": sex})
    df["area"]       = "all"
    df["year_start"] = 2010
    df["year_end"]   = 2010
    model_data.input_data = df

    # 1) 모델 생성 및 값 출력
    with pm.Model() as model:
        variables = dismod_mr.model.covariates.mean_covariate_model(
            data_type="test",
            mu=1.0,
            input_data=model_data.input_data,
            parameters=params,
            model=model_data,
            root_area="all",
            root_sex="total",
            root_year="all",
        )

        # 2) beta 리스트와 각 노드 정보 출력
        print("\n=== All beta nodes returned by build_alpha ===")
        for i, rv in enumerate(variables["beta"]):
            op_name = type(rv.owner.op).__name__ if hasattr(rv, "owner") else "NoOwner"
            print(f"  [{i}] name = {rv.name}, op = {op_name}")

        # 3) 모델 내 모든 Potential 정보 출력
        print("\n=== All Potentials in the model ===")
        for pot in model.potentials:
            pot_name = getattr(pot, "name", "<no name>")
            pot_op = pot.owner.op if hasattr(pot, "owner") else None
            op_name = type(pot_op).__name__ if pot_op is not None else "UnknownOp"
            print(f"  Potential name = {pot_name}, op = {op_name}")

        # 4) 특정 Potential("beta_test_x_sex_trunc") 존재 여부 출력
        pot_names = {getattr(pot, "name", None) for pot in model.potentials}
        print("\nPotential names set:", pot_names)
        assert "beta_test_x_sex_trunc" in pot_names, (
            "Expected a Potential named 'beta_test_x_sex_trunc' when using TruncatedNormal for x_sex."
        )

    # 5) 반환된 beta 변수에 owner가 있는지 출력
    beta_rv = variables["beta"][0]
    print(f"\nSelected beta_rv name: {beta_rv.name}")
    print("beta_rv has owner?", hasattr(beta_rv, "owner"))
    if hasattr(beta_rv, "owner"):
        print("beta_rv.operator:", type(beta_rv.owner.op).__name__)
        print("beta_rv.owner.inputs:")
        for i, inp in enumerate(beta_rv.owner.inputs):
            print(f"  input[{i}]: {inp} (type={type(inp)})")

    # 6) 추가 확인: TensorVariable인지 여부
    assert hasattr(beta_rv, "owner"), "beta_rv should be a TensorVariable with an owner"

    # 이제 디버깅을 위해 필요한 정보는 모두 출력되었습니다.

def test_random_effect_priors():
    model_data = dismod_mr.data.ModelData()
    hierarchy, output_template = dismod_mr.testing.data_simulation.small_output()
    model_data.hierarchy, model_data.output_template = hierarchy, output_template

    params = {
        "random_effects": {
            "USA": {"dist": "Normal", "mu": 0.1, "sigma": 0.5}
        }
    }

    n = 32
    area = np.random.choice(["all", "USA", "CAN"], size=n, p=[0.3, 0.3, 0.4])
    alpha_true = {"all": 0.0, "USA": 0.1, "CAN": -0.2}
    pi_true = np.exp([alpha_true[a] for a in area])
    sigma_true = 0.05
    p = np.random.normal(pi_true, sigma_true)

    df = pd.DataFrame({"value": p, "area": area})
    df["sex"]        = "male"
    df["year_start"] = 2010
    df["year_end"]   = 2010
    model_data.input_data = df

    # 1) 모델 생성
    with pm.Model() as model:
        variables = dismod_mr.model.covariates.mean_covariate_model(
            data_type="test",
            mu=1.0,
            input_data=model_data.input_data,
            parameters=params,
            model=model_data,
            root_area="all",
            root_sex="total",
            root_year="all",
        )

    # 2) variables["alpha"] 리스트 정보 출력
    print("\n=== All alpha nodes returned by build_alpha ===")
    for i, rv in enumerate(variables["alpha"]):
        print(f"  [{i}] name = {rv.name}, op = {type(rv.owner.op).__name__}")

    # 3) "USA" 노드가 실제로 리스트 몇 번째에 있는지 확인
    alpha_names = [rv.name for rv in variables["alpha"]]
    print("\nAlpha names list:", alpha_names)
    if "alpha_test_USA" not in alpha_names:
        print(">>> 'alpha_test_USA' was overridden by sum-to-zero deterministic! <<<")
    else:
        idx = alpha_names.index("alpha_test_USA")
        print(f"Found 'alpha_test_USA' at index {idx} in variables['alpha']")

    # 4) sum-to-zero으로 덮여 쓰인 경우, 원래 NormalRV가 들어있는 노드 탐색
    normal_nodes = []
    for rv in variables["alpha"]:
        op = rv.owner.op
        op_name = type(op).__name__
        # PyMC5에서 NormalRV 클래스는 RandomVariable의 서브클래스
        # 따라서 MRO를 탐색해 RandomVariable이 포함되었는지 확인
        is_random = any(
            base.__name__ == "RandomVariable"
            for base in op.__class__.__mro__
        )
        print(f"Checking rv {rv.name}: op={op_name}, is_random={is_random}")
        if rv.name == "alpha_test_USA" and is_random:
            normal_nodes.append(rv)

    print("\n=== Matched NormalRV nodes named 'alpha_test_USA' ===")
    print(normal_nodes)

    # 5) 최종적으로 사용할 alpha_rv 선택
    if len(normal_nodes) == 1:
        alpha_rv = normal_nodes[0]
        print(f"Using alpha_rv = {alpha_rv.name}, op = {type(alpha_rv.owner.op).__name__}")
    else:
        # sum-to-zero로 덮였거나, 예상과 다르게 생성됐을 수도 있음
        print(">>> Warning: Found", len(normal_nodes), "candidates for 'alpha_test_USA' <<<")
        if not normal_nodes:
            raise AssertionError("No NormalRV found for alpha_test_USA")
        alpha_rv = normal_nodes[0]

    # 6) mu 파라미터가 inputs[3]에 들어있는지 다시 확인
    inputs = alpha_rv.owner.inputs
    print(f"\nalpha_rv.owner.inputs (length={len(inputs)}):")
    for i, inp in enumerate(inputs):
        print(f"  inputs[{i}]: {inp}  (type={type(inp)})")

    # In PyMC5, inputs[3] is the mu for NormalRV
    mu_tensor = inputs[3]
    print("\nmu_tensor:", mu_tensor)

    # 7) mu_tensor 값 추출 시도
    try:
        mu_val = mu_tensor.get_value()
        print("mu_tensor.get_value() →", mu_val, type(mu_val))
    except AttributeError:
        mu_val = mu_tensor.eval()
        print("mu_tensor.eval()      →", mu_val, type(mu_val))

    # 8) numpy array로 바꾼 뒤 값 확인
    arr = np.array(mu_val)
    print("numpy array mu_val:", arr, "shape:", arr.shape)
    assert np.allclose(arr, 0.1), "Expected every entry of mu to be 0.1 for alpha_test_USA"





def test_covariate_model_dispersion():
    n = 100
    model_data = dismod_mr.data.ModelData()
    model_data.hierarchy, model_data.output_template = dismod_mr.testing.data_simulation.small_output()
    Z = np.random.randint(0,2,size=n)
    pi_true = 0.1; ess = 10000. * np.ones(n)
    eta_true = np.log(50); delta_true = 50 + np.exp(eta_true)
    p = np.random.negative_binomial(pi_true*ess, delta_true*np.exp(Z*(-0.2))) / ess

    df = pd.DataFrame({'value': p, 'z_0': Z})
    df['area'] = 'all'; df['sex'] = 'total'; df['year_start'] = 2000; df['year_end'] = 2000
    model_data.input_data = df

    with pm.Model():
        variables = dismod_mr.model.covariates.mean_covariate_model(
            data_type='test',
            mu=1,
            input_data=model_data.input_data,
            parameters={},
            model=model_data,
            root_area='all',
            root_sex='total', 
            root_year='all'
        )
        variables.update(
            dismod_mr.model.covariates.dispersion_covariate_model(
                'test', model_data.input_data, 0.1, 10.0
            )
        )
        dismod_mr.model.likelihood.neg_binom(
            'test', variables['pi'], variables['delta'], df['value'], ess
        )
        trace = pm.sample(draws=2, tune=0, step=pm.Metropolis(), chains=1,
                          cores=1, progressbar=False, return_inferencedata=False)
    print(trace)


def test_covariate_model_shift_for_root_consistency():
    # simulate interval data and test root consistency shift
    n=50; sigma_true=0.025
    a=np.arange(0,100,1)
    pi_age_true=0.0001*(a*(100.-a)+100.)

    d = dismod_mr.data.ModelData()
    d.input_data = dismod_mr.testing.data_simulation.simulated_age_intervals(
        'p', n, a, pi_age_true, sigma_true
    )
    d.hierarchy, d.output_template = dismod_mr.testing.data_simulation.small_output()

    with pm.Model():
        vars1 = dismod_mr.model.process.age_specific_rate(
            d, 'p', 'all', 'total', 'all', None, None, None
        )
        vars2 = dismod_mr.model.process.age_specific_rate(
            d, 'p', 'all', 'male', 1990, None, None, None
        )
        pm.sample(draws=3, tune=0, step=pm.Metropolis(), chains=1,
                  cores=1, progressbar=False, return_inferencedata=False)

    pi_usa = dismod_mr.model.covariates.predict_for(
        d, d.parameters['p'], 'all', 'male', 1990,
        'USA', 'male', 1990, 0., vars2['p'], 0., np.inf
    )
    assert isinstance(pi_usa, float)


def test_predict_for():
    # generate minimal interval data
    n=5; sigma_true=0.025
    a=np.arange(0,100,1)
    pi_age_true=0.0001*(a*(100.-a)+100.)

    d = dismod_mr.data.ModelData()
    d.input_data = dismod_mr.testing.data_simulation.simulated_age_intervals(
        'p', n, a, pi_age_true, sigma_true
    )
    d.hierarchy, d.output_template = dismod_mr.testing.data_simulation.small_output()

    vars = dismod_mr.model.process.age_specific_rate(
        d, 'p', 'all', 'total', 'all', None, None, None
    )
    mu_age = vars['mu_age']
    d.parameters['p'] = {'fixed_effects': {}, 'random_effects': {node:{'dist':'Constant','mu':0,'sigma':1e-9}
        for node in d.hierarchy.nodes}}
    pred = dismod_mr.model.covariates.predict_for(
        d, d.parameters['p'], 'all', 'total', 'all', 'USA','male',1990, 0., vars['p'],0., np.inf
    )
    expected = float(np.mean(mu_age))
    assert np.allclose(pred, expected)


def test_predict_for_wo_data():
    # predict without fitting data
    d = dismod_mr.data.ModelData()
    d.hierarchy, d.output_template = dismod_mr.testing.data_simulation.small_output()

    with pm.Model():
        vars = dismod_mr.model.process.age_specific_rate(
            d, 'p', 'all','total','all', None, None, None
        )
        pm.sample(draws=1, tune=0, step=pm.Metropolis(), chains=1,
                  cores=1, progressbar=False, return_inferencedata=False)

    d.parameters.setdefault('p', {}).setdefault('random_effects', {})
    for node in ['USA','NAHI','super-region-1','all']:
        d.parameters['p']['random_effects'][node] = {'dist':'Constant','mu':0,'sigma':1e-9}

    pred1 = dismod_mr.model.covariates.predict_for(
        d, d.parameters['p'],'all','total','all','USA','male',1990,0.,vars['p'],0.,np.inf
    )
    assert isinstance(pred1, float)


def test_predict_for_wo_effects():
    n=5; sigma_true=0.025
    a=np.arange(0,100,1)
    pi_age_true=0.0001*(a*(100.-a)+100.)

    d = dismod_mr.data.ModelData()
    d.input_data = dismod_mr.testing.data_simulation.simulated_age_intervals(
        'p', n, a, pi_age_true, sigma_true
    )
    d.hierarchy, d.output_template = dismod_mr.testing.data_simulation.small_output()

    with pm.Model():
        vars = dismod_mr.model.process.age_specific_rate(
            d, 'p', 'NAHI','male',2005,None,None,None, include_covariates=False
        )
        pm.sample(draws=10,tune=0,step=pm.Metropolis(),chains=1,cores=1,progressbar=False,return_inferencedata=False)

    pred = dismod_mr.model.covariates.predict_for(
        d, d.parameters['p'],'NAHI','male',2005,'USA','male',1990,0.,vars['p'],0.,np.inf
    )
    mu_age = vars['mu_age']
    expected = float(np.mean(mu_age))
    assert np.allclose(pred, expected)


def test_predict_for_w_region_as_reference():
    # simulate interval data for non-root reference region
    n=5; sigma_true=0.025
    a=np.arange(0,100,1)
    pi_age_true=0.0001*(a*(100.-a)+100.)

    d = dismod_mr.data.ModelData()
    d.input_data = dismod_mr.testing.data_simulation.simulated_age_intervals(
        'p', n, a, pi_age_true, sigma_true
    )
    d.hierarchy, d.output_template = dismod_mr.testing.data_simulation.small_output()

    with pm.Model():
        vars = dismod_mr.model.process.age_specific_rate(
            d, 'p', 'NAHI','male',2005,None,None,None
        )
        pm.sample(draws=10,tune=0,step=pm.Metropolis(),chains=1,cores=1,progressbar=False,return_inferencedata=False)

    # zero random effects
    d.parameters.setdefault('p', {}).setdefault('random_effects', {})
    for node in ['USA','NAHI','super-region-1','all']:
        d.parameters['p']['random_effects'][node] = {'dist':'Constant','mu':0.0,'sigma':1e-9}

    # case 1: zeros
    pred1 = dismod_mr.model.covariates.predict_for(
        d, d.parameters['p'],'NAHI','male',2005,'USA','male',1990,0.,vars['p'],0.,np.inf
    )
    expected1 = float(np.mean(vars['mu_age']))
    assert np.allclose(pred1, expected1)

    # case 2: non-zero RE only
    for i,node in enumerate(['USA','NAHI','super-region-1','all']):
        d.parameters['p']['random_effects'][node]['mu']=(i+1)/10.0
    pred2 = dismod_mr.model.covariates.predict_for(
        d,d.parameters['p'],'NAHI','male',2005,'USA','male',1990,0.,vars['p'],0.,np.inf
    )
    expected2 = float(np.mean(vars['mu_age'] * np.exp(0.1)))
    assert np.allclose(pred2, expected2)

    # case 3: stochastic RE for CAN
    np.random.seed(12345)
    pred3 = dismod_mr.model.covariates.predict_for(
        d,d.parameters['p'],'NAHI','male',2005,'CAN','male',1990,0.,vars['p'],0.,np.inf
    )
    assert isinstance(pred3, float)
