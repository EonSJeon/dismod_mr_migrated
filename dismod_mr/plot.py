""" Module for DisMod-MR graphics"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pymc as pm
import arviz as az

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f0', '#ffff33']



def data_bars(df, style='book', color='black', label=None, max=500):
    """ Plot data bars

    :Parameters:
      - `df` : pandas.DataFrame with columns age_start, age_end, value
      - `style` : str, either book or talk
      - `color` : str, any matplotlib color
      - `label` : str, figure label
      - `max` : int, number of data points to display
    """
    bars = list(zip(df['age_start'], df['age_end'], df['value']))
    if len(bars) > max:
        bars = bars.sample(max)
    x, y = [], []
    for a0, a1, v in bars:
        x += [a0, a1, np.nan]
        y += [v, v, np.nan]
    if style == 'book':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=4, color=color, label=label)
    elif style == 'talk':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=0,
                 alpha=1.0, color=colors[2], linewidth=15, label=label)
    else:
        raise ValueError(f'Unrecognized style: {style}')


def my_stats(node):
    """ Convenience function to generate summary stats from traces """
    try:
        trace = node.trace()
        mean = np.mean(trace, axis=0)
        lower = np.percentile(trace, 2.5, axis=0)
        upper = np.percentile(trace, 97.5, axis=0)
        hpd = np.vstack((lower, upper)).T
        return {'mean': mean, '95% HPD interval': hpd}
    except Exception:
        val = getattr(node, 'value', None)
        arr = np.array(val) if val is not None else np.array([])
        single = arr if arr.size > 0 else np.array([val])
        hpd = np.vstack((single, single)).T
        return {'mean': single, '95% HPD interval': hpd}


def asr(model, t):
    """ plot age-standardized rate fit and model data """
    vars = model.vars
    ages = np.array(model.parameters['ages'])
    df = model.get_data(t)
    data_bars(df, color='grey', label=f'{t} data')
    knots = vars[t].get('knots', np.arange(len(ages)))
    stats = my_stats(vars[t]['mu_age'])
    plt.plot(ages, stats['mean'], 'k-', lw=2, label='Posterior')
    plt.plot(ages[knots], stats['95% HPD interval'][knots, 0], 'k--', label='95% HPD')
    plt.plot(ages[knots], stats['95% HPD interval'][knots, 1], 'k--')


def plot_fit(model,
             data_types=['i','r','f','p','rr','pf'],
             ylab=['PY','PY','PY','Percent (%)','','PY'],
             plot_config=(2,3),
             with_data=True,
             with_ui=True,
             emp_priors=None,
             posteriors=None,
             fig_size=(10,6)):
    """ plot results of a fit """
    assert len(data_types) == len(ylab), 'data_types and ylab must match'
    vars = model.vars
    plt.figure(figsize=fig_size)
    try:
        ages = np.array(vars['i']['ages'])
    except KeyError:
        ages = np.array(vars[data_types[0]]['ages'])
    for j, t in enumerate(data_types):
        plt.subplot(plot_config[0], plot_config[1], j+1)
        if with_data:
            data_bars(model.input_data[model.input_data['data_type']==t],
                      color='grey', label='Data')
        knots = vars[t].get('knots', np.arange(len(ages)))
        stats = None
        try:
            stats = my_stats(vars[t]['mu_age'])
            plt.plot(ages, stats['mean'], 'k-', lw=2, label='Posterior')
            if with_ui:
                plt.plot(ages[knots], stats['95% HPD interval'][knots,0], 'k--')
                plt.plot(ages[knots], stats['95% HPD interval'][knots,1], 'k--')
        except Exception:
            if t in vars and 'mu_age' in vars[t]:
                val = vars[t]['mu_age'].value
                plt.plot(ages, val, 'k-', lw=2)
        if posteriors and t in posteriors:
            plt.plot(ages, posteriors[t], color='b', lw=1)
        if emp_priors and (t, 'mu') in emp_priors:
            mu = (emp_priors[t,'mu'] + 1e-9)[::5]
            s  = (emp_priors[t,'sigma'] + 1e-9)[::5]
            plt.errorbar(ages[::5], mu,
                         yerr=[mu - np.exp(np.log(mu)-(s/mu+ .1)),
                               np.exp(np.log(mu)+(s/mu+.1))-mu],
                         color='grey', lw=1, capsize=0)
        plt.xlabel('Age (years)')
        plt.ylabel(ylab[j])
        plt.title(t)


def effects(model, data_type, figsize=(22, 17)):
    """ Plot random effects (alpha) and fixed effects (beta). """
    vars_ = model.vars[data_type]
    hierarchy = model.hierarchy

    plt.figure(figsize=figsize)
    for col_idx, (covariate, effect) in enumerate([('U', 'alpha'), ('X', 'beta')]):
        if covariate not in vars_ or effect not in vars_:
            continue

        cov_names = list(vars_[covariate].columns)
        eff_list = vars_[effect]  # expected to be a list

        plt.subplot(1, 2, col_idx + 1)
        plt.title(f'{effect}_{data_type}')

        # collect means and hdi intervals
        means = []
        lowers = []
        uppers = []
        for var in eff_list:
            # get the posterior samples
            trace = var.trace() if hasattr(var, 'trace') else np.array([])
            if trace.size:
                mean = np.mean(trace, axis=0)
                hdi = az.hdi(trace, hdi_prob=0.95)
                lower, upper = mean - hdi[:, 0], hdi[:, 1] - mean
            else:
                # fallback to point estimate
                mean = getattr(var, 'initval', getattr(var, 'value', 0.0))
                lower = upper = np.zeros_like(mean)
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)

        means = np.atleast_1d(means)
        lowers = np.atleast_1d(lowers)
        uppers = np.atleast_1d(uppers)
        y = np.arange(len(means))

        # plot error bars
        plt.errorbar(
            means, y,
            xerr=[lowers, uppers],
            fmt='s', mec='w', color=colors[1]
        )
        plt.axvline(0, color='k', linestyle='--')

        plt.yticks([])
        plt.xlabel(effect)
        plt.ylabel('')  # no label for y-axis ticks

        # annotate covariate names, indented by hierarchy depth
        for yi, name in enumerate(cov_names):
            if name in hierarchy:
                depth = len(nx.shortest_path(hierarchy, 'all', name)) - 1
            else:
                depth = 0
            plt.text(
                means.min(), yi,
                ' ' + ('* ' * depth) + name,
                va='center', ha='left'
            )
        plt.ylim(-0.5, len(means) - 0.5)


def plot_hists(vars_dict):
    """ Plot histograms for all stochastic variables in a (nested) vars dict. """
    def plot_trace_hist(trace):
        plt.hist(trace, histtype='stepfilled', density=True)
        plt.yticks([])
        ticks = plt.xticks()[0]
        # show a few ticks for readability
        if len(ticks) >= 6:
            plt.xticks(ticks[1:6:2], fontsize=8)

    def recurse(v):
        if isinstance(v, dict):
            for sub in v.values():
                recurse(sub)
        elif hasattr(v, 'trace'):
            trace = v.trace()
            if trace.size:
                plot_trace_hist(trace)

    recurse(vars_dict)
    plt.tight_layout()

def plot_viz_of_stochs(vars_dict, viz_func, figsize=(8, 6)):
    """ Plot something (autocorr, trace, hist, etc.) for every free RV in ModelVars """
    plt.figure(figsize=figsize)
    # count how many subplots we need
    cells, stochs = tally_stochs(vars_dict)
    rows = int(np.floor(np.sqrt(cells)))
    cols = int(np.ceil(cells / rows)) if rows else 1

    tile = 1
    for rv in sorted(stochs, key=lambda v: v.name):
        # get its posterior trace array
        try:
            arr = rv.trace()
        except AttributeError:
            continue  # no trace
        arr = np.atleast_2d(arr)
        # for each dimension (in case vector-valued)
        for dim in range(arr.shape[1]):
            plt.subplot(rows, cols, tile)
            viz_func(arr[:, dim])
            plt.title(f"{rv.name}[{dim}]", fontsize=8, pad=10)
            tile += 1


def tally_stochs(vars_dict):
    """ Count all unobserved stochastic variables with non-empty traces """
    cells = 0
    stochs = []
    for v in vars_dict.values():
        # vars_dict may nest dicts
        seq = v.values() if isinstance(v, dict) else [v]
        for item in seq:
            if isinstance(item, list):
                seq.extend(item)
        for node in seq:
            if hasattr(node, "trace") and not getattr(node, "observed", False):
                tr = node.trace()
                if tr is not None and tr.size:
                    stochs.append(node)
                    cells += np.atleast_1d(node.value).size
    return cells, stochs


def plot_acorr(model):
    """ Autocorrelation plots for every stochastic in model.vars """
    from matplotlib import mlab

    def acorr_fn(x):
        if x.size > 50:
            plt.acorr(x, detrend=mlab.detrend_mean, maxlags=50, usevlines=True)
        plt.xticks([])
        plt.yticks([])
        l, r, b, t = plt.axis()
        plt.axis([-10, r, -0.1, 1.1])

    plot_viz_of_stochs(model.vars, acorr_fn, figsize=(12, 9))
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)


def plot_trace(model):
    """ Trace plots for every stochastic in model.vars """
    def trace_fn(x):
        plt.plot(x, linewidth=0.8)
        plt.xticks([])

    plot_viz_of_stochs(model.vars, trace_fn, figsize=(12, 9))
    plt.subplots_adjust(0.05, 0.01, 0.99, 0.99, 0.5, 0.5)


def plot_hists(vars_dict):
    """ Histogram for every stochastic in model.vars """
    def hist_fn(x):
        plt.hist(x, histtype='stepfilled', density=True)
        plt.yticks([])
        ticks = plt.xticks()[0]
        if len(ticks) >= 6:
            plt.xticks(ticks[1:6:2], fontsize=8)

    plot_viz_of_stochs(vars_dict, hist_fn, figsize=(8, 6))
    plt.tight_layout()


def data_value_by_covariates(inp: pd.DataFrame):
    """ Scatter of raw values vs each x_* covariate """
    X = inp.filter(like='x_')
    y = inp['value']
    n = len(X.columns)
    rows = int(np.ceil(n / 5))
    cols = min(n, 5)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, c in enumerate(X.columns):
        plt.subplot(rows, cols, i + 1)
        plt.title(c, fontsize=10)
        plt.scatter(X[c] + np.random.normal(scale=0.03, size=len(y)), y,
                    color='k', alpha=0.5, s=10)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    plt.tight_layout()


def plot_residuals(dm):
    """ Plot residuals for prevalence model """
    inp = dm.input_data
    pred = dm.vars['p']['mu_interval'].trace().mean(axis=0)
    ages = (inp['age_start'] + inp['age_end']) / 2 + np.random.randn(len(inp)) * 2
    res = inp['value'] - pred
    plt.figure()
    plt.scatter(ages, res, marker='s', edgecolor=colors[0], facecolor='white')
    plt.hlines(0, -5, 105, linestyle='dashed', color='gray')
    plt.xlabel('Age (years)')
    plt.ylabel('Residual (obs − pred)')
    a0 = dm.parameters['p']['parameter_age_mesh']
    for k in a0:
        plt.axvline(k, color=colors[1], linestyle='-')
    plt.xlim(-5, 105)
    plt.tight_layout()


def all_plots_for(model, t, ylab, emp_priors):
    """
    Convenience function: plot_fit, posterior predictive check,
    autocorr, and histograms for a single data type t.
    """
    from .plot import plot_fit, plot_one_ppc  # assume these live elsewhere
    plot_fit(model, data_types=[t], ylab=[ylab], plot_config=(1, 1), fig_size=(8, 8))
    plot_one_ppc(model, t)
    plot_acorr(model)
    plot_hists(model.vars)




