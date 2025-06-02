import networkx as nx
import numpy as np
import pandas as pd
import scipy.integrate
from dismod_mr.data import MRModel


def simulated_age_intervals(data_type, n, a, pi_age_true, sigma_true):
    # 1) pick random integer age intervals [start, end]
    age_start = np.random.uniform(0, 100, size=n).astype(int)
    age_start.sort()
    low = age_start + 1
    high = np.minimum(age_start + 10, 100)
    age_end = np.random.uniform(low, high, size=n).astype(int)

    # 2) true interval-average of pi_age_true using numpy.trapz
    pi_interval_true = [
        np.trapz(pi_age_true[s:e + 1], dx=1) / (e - s)
        for s, e in zip(age_start, age_end)
    ]

    # 3) covariates X and zero effects (beta_true all zeros)
    X = np.random.normal(0, 1, size=(n, 3))
    beta_true = [0, 0, 0]
    Y_true = X.dot(beta_true)

    # 4) apply covariate shift and add Gaussian noise with sd = sigma_true
    pi_true = np.array(pi_interval_true) * np.exp(Y_true)
    p = np.maximum(0.0, np.random.normal(loc=pi_true, scale=sigma_true))

    # 5) pack into a DataFrame
    df = pd.DataFrame({
        'value': p,
        'age_start': age_start,
        'age_end': age_end,
        'x_0': X[:, 0],
        'x_1': X[:, 1],
        'x_2': X[:, 2],
    })
    # effective sample size from variance formula, at least 1
    df['effective_sample_size'] = np.maximum(p * (1 - p) / sigma_true**2, 1.0)

    # fill in the rest of the columns to match ModelData expectations
    df['standard_error'] = np.nan
    df['upper_ci'] = np.nan
    df['lower_ci'] = np.nan
    df['year_start'] = 2005.0
    df['year_end'] = 2005.0
    df['sex'] = 'total'
    df['area'] = 'all'
    df['data_type'] = data_type

    return df


def small_output():
    # build a toy two-level hierarchy
    hierarchy = nx.DiGraph()
    hierarchy.add_node('all')
    hierarchy.add_edge('all', 'super-region-1', weight=0.1)
    hierarchy.add_edge('super-region-1', 'NAHI', weight=0.1)
    hierarchy.add_edge('NAHI', 'CAN', weight=0.1)
    hierarchy.add_edge('NAHI', 'USA', weight=0.1)

    # and a small output_template covering two areas, years, sexes
    output_template = pd.DataFrame({
        'year': [1990, 1990, 2005, 2005, 2010, 2010] * 2,
        'sex': ['male', 'female'] * 3 * 2,
        'x_0': [0.5] * 6 * 2,
        'x_1': [0.0] * 6 * 2,
        'x_2': [0.5] * 6 * 2,
        'pop': [50.0] * 6 * 2,
        'area': ['CAN'] * 6 + ['USA'] * 6,
    })

    return hierarchy, output_template


def simple_model(N):
    model = MRModel()
    # overwrite input_data with N empty rows
    model.input_data = pd.DataFrame(index=range(N))
    initialize_input_data(model.input_data)
    return model


def initialize_input_data(input_data):
    input_data['age_start'] = 0
    input_data['age_end'] = 1
    input_data['year_start'] = 2005.0
    input_data['year_end'] = 2005.0
    input_data['sex'] = 'total'
    input_data['data_type'] = 'p'
    input_data['standard_error'] = np.nan
    input_data['upper_ci'] = np.nan
    input_data['lower_ci'] = np.nan
    input_data['area'] = 'all'


def add_quality_metrics(df):
    df['abs_err'] = df['true'] - df['mu_pred']
    # relative error normalized by the crude mean of the true values
    df['rel_err'] = (df['true'] - df['mu_pred']) / df['true'].mean()
    df['covered?'] = ((df['true'] >= df['lb_pred'])
                      & (df['true'] <= df['ub_pred']))


def initialize_results(model):
    # prepare empty lists to collect scalars
    model.results = {
        'param': [],
        'bias': [],
        'rel_bias': [],
        'mae': [],
        'mare': [],
        'pc': [],
        'time': [],
    }


def finalize_results(model):
    # convert the dict of lists into a DataFrame, in this exact column order
    cols = 'param bias rel_bias mae mare pc time'.split()
    model.results = pd.DataFrame(model.results, columns=cols)


def add_to_results(model, name):
    df = getattr(model, name)
    model.results['param'].append(name)
    model.results['bias'].append(df['abs_err'].mean())
    model.results['rel_bias'].append(df['rel_err'].mean())
    model.results['mae'].append(np.median(np.abs(df['abs_err'].dropna())))
    model.results['mare'].append(np.median(np.abs(df['rel_err'].dropna())))
    model.results['pc'].append(df['covered?'].mean())
    model.results['time'].append(model.mcmc.wall_time)
