"""Test age group models."""
import numpy as np
import pymc as pm
import arviz as az

import dismod_mr


def test_age_standardizing_approx():
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
        # age standardizing (approximation)
        age_weights = np.ones_like(ages)
        variables.update(
            dismod_mr.model.age_groups.age_standardize_approx(
                'test',
                age_weights,
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
            sigma=0,
            p=d['value'],
            s=sigma_true
        )

        step = pm.Metropolis()            # ← NUTS 대신 Metropolis
        trace = pm.sample(
            draws=3,
            tune=0,                       # Metropolis는 tune=0 으로도 OK
            step=step,
            chains=1,
            cores=1,
            progressbar=False,
            return_inferencedata=False
        )
        print(trace)


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
