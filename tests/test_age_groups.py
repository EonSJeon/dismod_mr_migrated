"""Test age group models."""
import numpy as np
import pymc as pm

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

        # minimal sampling for test
        pm.sample(draws=3, chains=1, tune=0, cores=1, progressbar=False)


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
            sigma=0,
            p=d['value'],
            s=sigma_true
        )

        # minimal sampling for test
        pm.sample(draws=3, chains=1, tune=0, cores=1, progressbar=False)
