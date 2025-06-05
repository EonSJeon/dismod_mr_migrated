import pandas as pd
import numpy as np
import pymc as pm
import dismod_mr
import numba
import networkx as nx
import json
import re
import os
from typing import Dict, Any
import pytensor.tensor as at

def my_stats(self, alpha: float = 0.05, start: int = 0, batches: int = 100,
             chain=None, quantiles=(2.5, 25, 50, 75, 97.5)) -> dict:
    """
    Compute basic summary statistics and HDI for a PyMC trace.
    """
    trace = self.trace()
    n = len(trace)
    if n == 0:
        print(f'Cannot generate statistics for zero-length trace in {self.__name__}')
        return {}

    # highest-density interval using ArviZ-backed PyMC API
    hdi = pm.stats.hdi(trace, hdi_prob=1 - alpha)
    return {
        'n': n,
        'standard deviation': trace.std(axis=0),
        'mean': trace.mean(axis=0),
        f'{int(100*(1-alpha))}% HDI': hdi,
    }


def describe_vars(vars_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Describe a collection of PyMC/Aesara nodes:
    - Python type
    - a representative numeric value (if available)
    - whether a .logp method exists
    """
    df = pd.DataFrame(
        columns=['type', 'value', 'logp_exists'],
        index=list(vars_dict.keys()),
        dtype=object
    )

    for name, node in vars_dict.items():
        # 1) Record Python type
        df.loc[name, 'type'] = type(node).__name__

        # 2) Attempt to extract a representative numeric value
        display_val = None

        # 2a) If this is a pandas DataFrame or Series, report its shape and skip eval
        if isinstance(node, (pd.DataFrame, pd.Series)):
            display_val = f"<{type(node).__name__} shape={node.shape}>"

        # 2b) If this is an Aesara TensorVariable, call .eval()
        elif isinstance(node, at.TensorVariable):
            try:
                val = node.eval()
                if isinstance(val, np.ndarray):
                    display_val = float(val.flat[0]) if val.size == 1 else f"{val.flat[0]:.3g}, …"
                else:
                    display_val = val
            except Exception:
                display_val = "<eval failed>"

        else:
            # 2c) If this object has an 'initval' attribute (PyMC RandomVariable), use that
            if hasattr(node, 'initval'):
                iv = node.initval
                if isinstance(iv, np.ndarray):
                    display_val = float(iv.flat[0]) if iv.size == 1 else f"{iv.flat[0]:.3g}, …"
                else:
                    display_val = iv

            # 2d) If this is pm.Data or another object with get_value(), use get_value()
            elif hasattr(node, 'get_value'):
                try:
                    val = node.get_value()
                    if isinstance(val, np.ndarray):
                        display_val = float(val.flat[0]) if val.size == 1 else f"{val.flat[0]:.3g}, …"
                    else:
                        display_val = val
                except Exception:
                    display_val = "<get_value failed>"

            # 2e) Otherwise, if it has a plain 'value' attribute, show that (but ensure scalar)
            elif hasattr(node, 'value'):
                val = getattr(node, 'value')
                if isinstance(val, np.ndarray):
                    display_val = float(val.flat[0]) if val.size == 1 else f"{val.flat[0]:.3g}, …"
                else:
                    display_val = val

            else:
                display_val = "<non-numeric>"

        df.loc[name, 'value'] = display_val

        # 3) Check for presence of a logp method (PyMC 5 uses .logp for distributions)
        df.loc[name, 'logp_exists'] = hasattr(node, 'logp')

    return df.sort_values(by='type')


def check_convergence(vars_dict: dict) -> bool:
    """
    Simple convergence check: for each stochastic, compare lagged autocorrelation.
    Warn if any autocorr > 0.5 at lags 50–99.
    """
    # obtain list of stochastics via plotting utility
    _, stochs = dismod_mr.plot.tally_stochs(vars_dict)
    for s in sorted(stochs, key=lambda x: x.name if hasattr(x, 'name') else x.__name__):
        tr = s.trace()
        if tr.ndim == 1:
            tr = tr.reshape(-1, 1)
        # center once
        tr_centered = tr - tr.mean(axis=0)
        for lag in range(50, 100):
            if lag >= tr_centered.shape[0]:
                break
            num = np.sum(tr_centered[:-lag] * tr_centered[lag:], axis=0)
            den = np.sum(tr_centered[lag:] ** 2, axis=0)
            ac = num / den
            if np.any(np.abs(ac) > 0.5):
                print(f'Potential non-convergence in {s.name if hasattr(s, "name") else s}: autocorr {ac}')
                return False
    return True

class ModelVars(dict):
    """
    Container for PyMC variables making up the model.

    Supports:
    - dict-like access
    - += for adding new variables
    - describe() to print a summary
    - empirical_priors_from_fit() to extract constant priors from fitted vars
    """
    def __iadd__(self, other: dict):
        self.update(other)
        return self

    def __str__(self):
        df = describe_vars(self)
        keys = ', '.join(self.keys())
        return f"{df} keys: {keys}"

    def describe(self):
        print(describe_vars(self))

    def empirical_priors_from_fit(self, type_list=['i', 'r', 'f', 'p', 'rr']):
        """
        Generate empirical constant priors based on fitted random and fixed effects.
        Returns a dict with random_effects and fixed_effects for each rate type.
        """
        prior_dict = {}
        for t in type_list:
            if t in self:
                entry = self[t]
                pdt = {'random_effects': {}, 'fixed_effects': {}}
                # random effects
                if 'U' in entry and 'alpha' in entry:
                    for idx, re in enumerate(entry['U'].columns):
                        alpha_var = entry['alpha'][idx]
                        if hasattr(alpha_var, 'trace'):
                            mu = my_stats(alpha_var)['mean']
                        else:
                            mu = float(alpha_var)
                        pdt['random_effects'][re] = {'dist': 'Constant', 'mu': mu}
                # fixed effects
                if 'X' in entry and 'beta' in entry:
                    for idx, fe in enumerate(entry['X'].columns):
                        beta_var = entry['beta'][idx]
                        if hasattr(beta_var, 'trace'):
                            mu = my_stats(beta_var)['mean']
                        else:
                            mu = float(beta_var)
                        pdt['fixed_effects'][fe] = {'dist': 'Constant', 'mu': mu}
                prior_dict[t] = pdt
        return prior_dict

class MRModel:
    """
    Holds input data, parameters, hierarchy, and model variables.
    Provides loading, filtering, describing, plotting, fitting, and saving.
    """
    def __init__(self):
        self.input_data = pd.DataFrame(columns=[
            'data_type','value','area','sex','age_start','age_end',
            'year_start','year_end','standard_error',
            'effective_sample_size','lower_ci','upper_ci','age_weights'
        ])
        self.output_template = pd.DataFrame(columns=['data_type','area','sex','year','pop'])
        self.parameters = dict(i={}, p={}, r={}, f={}, rr={}, X={}, pf={}, ages=list(range(101)))
        self.hierarchy = nx.DiGraph()
        self.hierarchy.add_node('all')
        self.nodes_to_fit = list(self.hierarchy.nodes())
        self.vars = ModelVars()
        self.model_settings = {}

    def get_data(self, data_type: str) -> pd.DataFrame:
        if not self.input_data.empty:
            return self.input_data[self.input_data['data_type'] == data_type]
        return self.input_data

    def keep(self,
             areas: list = ['all'],
             sexes: list = ['male','female','total'],
             start_year: int = -np.inf,
             end_year: int = np.inf):
        if 'all' not in areas:
            self.hierarchy.remove_node('all')
            for area in areas:
                self.hierarchy.add_edge('all', area)
            self.hierarchy = nx.bfs_tree(self.hierarchy, 'all')
            relevant = self.input_data['area'].isin(self.hierarchy.nodes()) | (self.input_data['area']=='all')
            self.input_data = self.input_data.loc[relevant]
            self.nodes_to_fit = list(set(self.hierarchy.nodes()) & set(self.nodes_to_fit))
        self.input_data = self.input_data[self.input_data['sex'].isin(sexes)]
        self.input_data = self.input_data[self.input_data['year_end'] >= start_year]
        self.input_data = self.input_data[self.input_data['year_start'] <= end_year]
        print(f'kept {len(self.input_data)} rows of data')

    @staticmethod
    def load(path: str) -> 'MRModel':
        def load_jsonc(fp):
            txt = open(fp, encoding='utf-8').read()
            no_comments = re.sub(r'//.*?$|/\*.*?\*/', '', txt, flags=re.MULTILINE|re.DOTALL)
            clean = re.sub(r',\s*([}\]])', r'\1', no_comments)
            return json.loads(clean)
        def load_any(name):
            for ext, loader in (('.jsonc', load_jsonc), ('.json', lambda fp: json.load(open(fp, encoding='utf-8')))):
                fp = os.path.join(path, f"{name}{ext}")
                if os.path.isfile(fp):
                    return loader(fp)
            raise FileNotFoundError(f"No {name}.json(c) in {path}")
        d = MRModel()
        d.input_data = pd.read_csv(os.path.join(path, 'input_data.csv'))
        d.output_template = pd.read_csv(os.path.join(path, 'output_template.csv'))
        params = load_any('parameters')
        hier = load_any('hierarchy')
        d.parameters = params
        d.hierarchy = nx.DiGraph()
        d.hierarchy.add_nodes_from(hier['nodes'])
        d.hierarchy.add_edges_from(hier['edges'])
        d.nodes_to_fit = load_any('nodes_to_fit')
        return d

    def describe(self, data_type: str):
        G = self.hierarchy
        df = self.get_data(data_type)
        for n in nx.dfs_postorder_nodes(G, 'all'):
            cnt = df['area'].eq(n).sum() + sum(G.nodes[c].get('cnt', 0) for c in G.successors(n))
            G.nodes[n]['cnt'] = int(cnt)
            G.nodes[n]['depth'] = nx.shortest_path_length(G, 'all', n)
        for n in nx.dfs_preorder_nodes(G, 'all'):
            if G.nodes[n]['cnt'] > 0:
                print('  '*G.nodes[n]['depth'] + n, G.nodes[n]['cnt'])

    def plot(self, rate_type=None):
        import matplotlib.pyplot as plt
        import numpy as _np
        import dismod_mr.plot as plot
        types = rate_type or self.model_settings.get('rate_type', ['p','i','r','f'])
        fig = plt.figure()
        for i, t in enumerate(types):
            if len(types)==4:
                plt.subplot(2,2,i+1)
                plt.title(t)
                plt.subplots_adjust(hspace=.5,wspace=.4)
            plot.data_bars(self.get_data(t), color=plot.colors[1])
            if t in self.vars:
                x = _np.array(self.parameters['ages'])
                knots = self.vars[t].get('knots', _np.array([]))
                mu_node = self.vars[t].get('mu_age')
                if not hasattr(mu_node, 'trace'):
                    pt = mu_node.value
                    plt.plot(x, pt, lw=3, color=plot.colors[0])
                else:
                    pred = mu_node.trace()
                    ui = pm.stats.hdi(pred, hdi_prob=0.95)
                    plt.fill_between(x, ui[:,0], ui[:,1], color=plot.colors[0], alpha=0.3)
                    plt.plot(x, pred.mean(0), lw=2, color=plot.colors[0])
            plt.axis(xmin=-5, xmax=105)

    def invalid_precision(self) -> pd.DataFrame:
        mask = (self.input_data.effective_sample_size.isnull() &
                self.input_data.standard_error.isnull() &
                (self.input_data.lower_ci.isnull() | self.input_data.upper_ci.isnull()))
        return self.input_data[mask]

    def fit(self, how='mcmc', iter=10000, burn=5000, thin=5):
        from . import fit
        if 'rate_type' in self.model_settings:
            rt = self.model_settings['rate_type']
            if how=='mcmc':
                self.map, self.mcmc = fit.asr(self, rt, iter=iter, burn=burn, thin=thin)
            else:
                self.map = pm.find_MAP(model=self.vars[rt])
        elif 'consistent' in self.model_settings:
            if how=='mcmc':
                self.map, self.mcmc = fit.consistent(self, iter=iter, burn=burn, thin=thin)
            else:
                raise NotImplementedError
        else:
            raise RuntimeError('Call setup_model before fit')

    def predict_for(self, rate_type, area, sex, year):
        return dismod_mr.model.covariates.predict_for(
            model=self,
            parameters=self.parameters[rate_type],
            root_area='all', root_sex='total', root_year='all',
            area=area, sex=sex, year=year,
            population_weighted=True,
            vars=self.vars[rate_type],
            lower=self.parameters[rate_type].get('level_bounds',{}).get('lower', 0),
            upper=self.parameters[rate_type].get('level_bounds',{}).get('upper', 1)
        )

    def set_smoothness(self, rate_type, value):
        """ Set smoothness parameter for age-specific rate function of one type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf', 'm', 'X', or 'csmr'
          - `value` : str, one of 'No Prior', 'Slightly', 'Moderately', or 'Very',
            or non-negative float

        :Results:
          - Changes the smoothing parameter in self.parameters

        """
        self.parameters[rate_type]['smoothness'] = dict(age_start=0, age_end=100, amount=value)

    def set_knots(self, rate_type, value):
        """ Set knots for age-specific rate function of one type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `value` : list, positions knots, start and end must
            correspond to parameters['ages']

        :Results:
          - Changes the knots in self.parameters[rate_type]

        """
        self.parameters[rate_type]['parameter_age_mesh'] = value

    def set_level_value(self, rate_type, age_before=None, age_after=None, value=0):
        """ Set level value for age-specific rate function of one
        type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `age_before` : int, level value is applied for all ages
            less than this
          - `age_after` : int, level value is applied for all ages
            more than this
          - `value` : float, value of the age-specific rate function
            before and after specified ages

        :Results:
          - Changes level_value in self.parameters[rate_type]

        """
        if age_before == None:
            age_before = 0
        if age_after == None:
            age_after = 0

        self.parameters[rate_type]['level_value'] = dict(age_before=age_before, age_after=age_after, value=value)
        if 'level_bounds' not in self.parameters[rate_type]:
            self.set_level_bounds(rate_type, lower=0, upper=1)  # level bounds are needed for level value prior to work

    def set_level_bounds(self, rate_type, lower=0, upper=1):
        """ Set level bounds for age-specific rate function of one
        type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `lower` : float, minimum value of the age-specific rate
            function
          - `upper` : float, maximum value of the age-specific rate
            function


        :Results:
          - Changes level_bounds in self.parameters[rate_type]

        """
        self.parameters[rate_type]['level_bounds'] = dict(lower=lower, upper=upper)
        if 'level_value' not in self.parameters[rate_type]:
            self.set_level_value(rate_type, age_before=-1, age_after=101)  # level values are needed for level value prior to work

    def set_increasing(self, rate_type, age_start, age_end):
        """ Set increasing prior for age-specific rate function of one
        type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `age_start` : int, minimum age of the age-specific rate
            function prior to increase
          - `age_end` : int, maximum age of the age-specific rate
            function prior to increase


        :Results:
          - Changes increasing in self.parameters[rate_type]

        """
        self.parameters[rate_type]['increasing'] = dict(age_start=age_start, age_end=age_end)

    def set_decreasing(self, rate_type, age_start, age_end):
        """ Set decreasing prior for age-specific rate function of one
        type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p', 'rr', 'pf',
            'm', 'X', or 'csmr'
          - `age_start` : int, minimum age of the age-specific rate
            function prior to decrease
          - `age_end` : int, maximum age of the age-specific rate
            function prior to decrease


        :Results:
          - Changes decreasing in self.parameters[rate_type]

        """
        self.parameters[rate_type]['decreasing'] = dict(age_start=age_start, age_end=age_end)

    def set_heterogeneity(self, rate_type, value):
        """ Set heterogeneity prior for age-specific rate function of one
        type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p'
          - `value` : str, one of 'Unusable', 'Very', 'Moderately', or
            'Slightly'


        :Results:
          - Changes heterogeneity in self.parameters[rate_type]

        """
        self.parameters[rate_type]['heterogeneity'] = value

    def set_effect_prior(self, rate_type, cov, value):
        """ Set prior for fixed or random effect of one
        type.

        :Parameters:
          - `rate_type` : str, one of 'i', 'r', 'f', 'p'
          - `cov` : str, covariate name
          - `value` : dict, including keys `dist`, `mu`, and possibly
            `sigma`, `lower`, and `upper`

        :Results:
          - Changes heterogeneity in self.parameters[rate_type]

        :Notes:

        the `value` dict describes the distribution of the effect
        prior.  Recognized distributions are Constant, Normal, and
        TruncatedNormal.  Examples:
          - `value=dict(dist='Constant', mu=0)`
          - `value=dict(dist='Normal', mu=0, sigma=1)`
          - `value=dict(dist='TruncatedNormal, mu=0, sigma=1,
                        lower=-1, upper=1)`
        """
        for effects in ['fixed_effects', 'random_effects']:
            if not effects in self.parameters[rate_type]:
                self.parameters[rate_type][effects] = {}

        if cov.startswith('x_'): # fixed effect
            self.parameters[rate_type]['fixed_effects'][cov] = value
        else: # random effect
            self.parameters[rate_type]['random_effects'][cov] = value


    def setup_model(self, rate_type=None, rate_model='neg_binom',
        interpolation_method='linear', include_covariates=True):
        """ Setup PyMC model vars based on current parameters and data
        :Parameters:
        - `rate_type` : str, optional if rate_type is provided, the
        model will be an age standardized rate model for the
        specified rate_type.  otherwise, it will be a consistent
        model for all rate types.
        - `rate_model` : str, optional, one of 'beta_binom',
        'beta_binom_2', 'binom', 'log_normal', 'neg_binom',
        'neg_binom_lower_bound', 'neg_binom', 'normal',
        'offest_log_normal', or 'poisson' if rate_type is
        provided, this option specifies the rate model for data of
        this rate type.
    
        - `interpolation_method` : str, optional, one of 'linear',
        'nearest', 'zero', 'slinear', 'quadratic, or 'cubic'
    
        - `include_covariates` : bool, optional if rate_type is
        provided, this option specifies if the model for the rate
        type should include additional fixed and random effects
    
        :Notes:
        This method also creates methods fit and predict_for for the
        current object
        """
    
        if rate_type:
            self.vars = model.asr(self, rate_type,
                              rate_type=rate_model,  # TODO: rename parameter in model.process.asr so this is less confusing
                              interpolation_method=interpolation_method,
                              include_covariates=include_covariates)
        
            self.model_settings['rate_type'] = rate_type
    
        else:
            self.vars = model.consistent(self, rate_type=rate_model)
            self.model_settings['consistent'] = True
    
    def save(self, path):
        """ Saves all model data in human-readable files
    
        :Parameters:
        - `path` : str, directory to save in
    
        :Results:
        - Saves files to specified path, overwritting what was there before
    
        """
    
        self.input_data.to_csv(path + '/input_data.csv')
        self.output_template.to_csv(path + '/output_template.csv')
        json.dump(self.parameters, open(path + '/parameters.json', 'w'), indent=2)
        json.dump(dict(nodes=[[n, self.hierarchy.node[n]] for n in sorted(self.hierarchy.nodes())],
               edges=[[u, v, self.hierarchy.edge[u][v]] for u,v in sorted(self.hierarchy.edges())]),
          open(path + '/hierarchy.json', 'w'), indent=2)
        json.dump(list(self.nodes_to_fit), open(path + '/nodes_to_fit.json', 'w'), indent=2)


load = MRModel.load