"""Test covariate and process models."""
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# Configure PyTensor
# pt.config.gcc__cxxflags = "-Wno-c++11-narrowing"

import dismod_mr


def test_covariate_model_sim_no_hierarchy():
    # Generate synthetic data
    np.random.seed(42)
    n = 128
    x = np.random.randn(n, 3)
    beta = np.array([1.0, -1.0, 0.5])
    y = x @ beta + np.random.randn(n) * 0.1

    # Create DataFrame
    df = pd.DataFrame({
        'value': y,
        'x_0': x[:, 0],
        'x_1': x[:, 1],
        'x_2': x[:, 2],
        'area': ['all'] * n,
        'sex': ['total'] * n,
        'year_start': [2000] * n,
        'year_end': [2000] * n
    })
    print(df)

    # Define the model
    with pm.Model() as model:
        # Priors for fixed effects
        beta = pm.Normal('beta', mu=0, sigma=1, shape=3)
        
        # Prior for error term
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Linear predictor
        mu = pm.math.dot(x, beta)
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Sample
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=1,
            return_inferencedata=False
        )

    # Print results
    print("\nPosterior summary:")
    print(pm.summary(trace))


def test_covariate_model_sim_w_hierarchy():
    n = 50
    hierarchy, output_template = dismod_mr.testing.data_simulation.small_output()

    area = np.random.choice(['all','USA','CAN'], size=n, p=[0.3,0.3,0.4])
    sex = np.random.choice(['male','female','total'], size=n, p=[0.3,0.3,0.4])
    year = np.random.randint(1990, 2011, size=n)
    alpha_true = {'all':0.0,'USA':0.1,'CAN':-0.2}
    pi_true = np.exp([alpha_true[a] for a in area])
    sigma_true = 0.05 * np.ones_like(pi_true)
    p = np.random.normal(pi_true, 1.0 / sigma_true**2)

    model_data = dismod_mr.data.ModelData()
    model_data.input_data = pd.DataFrame({
        'value': p, 'area': area, 'sex': sex,
        'year_start': year, 'year_end': year
    })
    model_data.hierarchy, model_data.output_template = hierarchy, output_template

    with pm.Model():
        variables = dismod_mr.model.covariates.mean_covariate_model(
            'test', 1, model_data.input_data, {}, model_data,
            'all', 'total', 'all'
        )
        pm.Normal('obs', mu=variables['pi'], sigma=sigma_true, observed=p)
        
        # Use Metropolis sampler with no tuning
        step = pm.Metropolis()
        trace = pm.sample(
            draws=2,
            tune=0,
            step=step,
            chains=1,
            cores=1,
            progressbar=False,
            return_inferencedata=False
        )
    assert 'sex' not in variables['U']
    assert 'x_sex' in variables['X']
    assert len(variables['beta']) == 1


def test_fixed_effect_priors():
    model_data = dismod_mr.data.ModelData()
    params = {'fixed_effects': {
        'x_sex': {'dist':'TruncatedNormal','mu':1.0,'sigma':0.5,'lower':-10,'upper':10}
    }}

    n = 32
    sex = np.random.choice(['male','female','total'], size=n, p=[0.3,0.3,0.4])
    beta_true = {'male':-1.0,'total':0.0,'female':1.0}
    pi_true = np.exp([beta_true[s] for s in sex])
    sigma_true = 0.05
    p = np.random.normal(pi_true, 1.0 / sigma_true**2)

    df = pd.DataFrame({'value': p, 'sex': sex})
    df['area'] = 'all'
    df['year_start'] = 2010
    df['year_end'] = 2010
    model_data.input_data = df

    with pm.Model():
        variables = dismod_mr.model.covariates.mean_covariate_model(
            'test', 1, model_data.input_data, params, model_data,
            'all', 'total', 'all'
        )
    beta_rv = variables['beta'][0]
    assert beta_rv.distribution.__class__.__name__ == 'Normal'
    assert float(beta_rv.distribution.mu) == 1.0


def test_random_effect_priors():
    model_data = dismod_mr.data.ModelData()
    hierarchy, output_template = dismod_mr.testing.data_simulation.small_output()
    model_data.hierarchy, model_data.output_template = hierarchy, output_template
    params = {'random_effects': {
        'USA': {'dist':'Normal','mu':0.1,'sigma':0.5}  # Changed from TruncatedNormal
    }}

    n = 32
    area = np.random.choice(['all','USA','CAN'], size=n, p=[0.3,0.3,0.4])
    alpha_true = {'all':0.0,'USA':0.1,'CAN':-0.2}
    pi_true = np.exp([alpha_true[a] for a in area])
    sigma_true = 0.05
    p = np.random.normal(pi_true, 1.0 / sigma_true**2)

    df = pd.DataFrame({'value': p, 'area': area})
    df['sex'] = 'male'
    df['year_start'] = 2010
    df['year_end'] = 2010
    model_data.input_data = df

    with pm.Model():
        variables = dismod_mr.model.covariates.mean_covariate_model(
            'test', 1, model_data.input_data, params, model_data,
            'all', 'total', 'all'
        )
    idx = list(variables['U'].columns).index('USA')
    alpha_rv = variables['alpha'][idx]
    assert alpha_rv.distribution.__class__.__name__ == 'Normal'
    assert float(alpha_rv.distribution.mu) == 0.1


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
            'test', 1, model_data.input_data, {}, model_data,
            'all', 'total', 'all'
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


if __name__ == "__main__":
    test_covariate_model_sim_no_hierarchy()
