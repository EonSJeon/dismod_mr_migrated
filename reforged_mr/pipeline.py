import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import json
import re
import networkx as nx
import logging
import time
import arviz as az
import warnings
import matplotlib.pyplot as plt
import random

import model.spline as spline
print(spline.__file__)
import model.priors as priors
print(priors.__file__)
import model.age_groups as age_groups
print(age_groups.__file__)
import model.covariates as covariates
print(covariates.__file__)
import model.likelihood as likelihood
print(likelihood.__file__)


################################################################################
#########################   HELPER FUNCTIONS   #################################
################################################################################


########### Three load functions to read json and jsonc files. #################
def load_jsonc(filepath):
    """Load JSONC file (JSON with comments)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove single-line comments (// ...)
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Remove trailing commas before closing brackets/braces
    content = re.sub(r',\s*([}\]])', r'\1', content)
    
    return json.loads(content)

def load_json(filepath):
    """Load regular JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_any(filepath):
    """Load either JSON or JSONC file based on extension"""
    if filepath.endswith('.jsonc'):
        return load_jsonc(filepath)
    else:
        return load_json(filepath)


########### visualize the hierarchy ##########################
def describe_hierarchy(model):
    G          = model.shared_data['region_id_graph']
    id_to_name = model.shared_data['id_to_name']

    depths = {n: nx.shortest_path_length(G, 1, n) for n in G.nodes}

    for n in nx.dfs_preorder_nodes(G, 1):
        indent = '  ' * depths[n]
        print(f"{indent}{id_to_name[n]} ({n})")

def describe_data(model):
    G               = model.shared_data['region_id_graph']
    data            = model.shared_data['data']
    id_to_name      = model.shared_data['id_to_name']
    
    for n in nx.dfs_postorder_nodes(G, 1):
        cnt = data['location_id'].eq(n).sum() + sum(G.nodes[c].get('cnt', 0) for c in G.successors(n))
        G.nodes[n]['cnt'] = int(cnt)
        G.nodes[n]['depth'] = nx.shortest_path_length(G, 1, n)
        
    for n in nx.dfs_preorder_nodes(G, 1):
        if G.nodes[n]['cnt'] > 0:
            print('  '*G.nodes[n]['depth'] + id_to_name[n] + f' ({n}): ', G.nodes[n]['cnt'])


########### inspect the model ###################################################
def inspect_model(model, var_name=None, show_shared_data=False):
    """
    Inspect a PyMC model. If var_name is None, print a summary,
    plus any shared_data contents. Otherwise, show details about a specific variable.
    """
    if var_name is None:
        print("📊 Model Summary:")
        print(f"  • Free RVs       : {len(model.free_RVs)} {[rv.name for rv in model.free_RVs]}")
        print(f"  • Observed RVs   : {len(model.observed_RVs)} {[rv.name for rv in model.observed_RVs]}")
        print(f"  • Deterministics : {len(model.deterministics)} {[rv.name for rv in model.deterministics]}")
        print(f"  • Potentials     : {len(model.potentials)} {[pot.name for pot in model.potentials]}")
        print(f"  • Total Named RVs: {len(model.named_vars)}")

        # --- Print shared_data contents if present ---
        if show_shared_data:
            if hasattr(model, "shared_data"):
                sd = model.shared_data
                if isinstance(sd, dict) and sd:
                    print("\n🔖 shared_data:")
                    for key, val in sd.items():
                        if isinstance(val, np.ndarray):
                            print(f"  • {key:15s}: array, shape={val.shape}, dtype={val.dtype}")
                        else:
                            print(f"  • {key:15s}: {val!r}")

    else:
        var_dict = model.named_vars
        if var_name not in var_dict:
            print(f"❌ Variable '{var_name}' not found in model.named_vars.")
            return

        var = var_dict[var_name]
        print(f"🔍 Variable: {var_name}")
        print(f"  • Type     : {type(var)}")
        print(f"  • Shape    : {getattr(var, 'shape', None)}")
        print(f"  • DType    : {getattr(var, 'dtype', None)}")
        print(f"  • Owner OP : {var.owner.op if getattr(var, 'owner', None) else 'None'}")

        if hasattr(var, 'distribution'):
            dist = var.distribution
            print(f"  • Distribution: {dist.__class__.__name__}")
            if hasattr(dist, 'dist'):
                print(f"    - PyMC Dist : {dist.dist.__class__.__name__}")
            if hasattr(dist, 'kwargs'):
                print("    - Parameters:")
                for k, v in dist.kwargs.items():
                    print(f"      {k}: {v}")

        if hasattr(var, 'eval'):
            try:
                val = var.eval()
                print(f"  • Current value (eval): {val}")
            except Exception as e:
                print(f"  • Could not evaluate variable: {e}")



########### check rhat condition ##########################################################
def return_rhat(idata):
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar divide",
        category=RuntimeWarning,
    )
    summary_df = az.summary(idata)
    total_vars = len(summary_df)
    over_1_01 = (summary_df["r_hat"] > 1.01).sum()
    under_1_01 = (summary_df["r_hat"] <= 1.01).sum()
    n_missing = summary_df["r_hat"].isna().sum()
    print(f"Total vars:    {total_vars}")
    print(f"R-hat > 1.01:  {over_1_01}")
    print(f"R-hat <= 1.01:  {under_1_01}")
    print(f"R-hat missing: {n_missing}")
    print(az.rhat(idata))
    return summary_df



########### visualize the data ###################################################
def data_bars(df, style='book', color='black', label=None, max=500):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f0', '#ffff33']
    bars = list(zip(df['age_start'], df['age_end'], df['value']))
    if len(bars) > max:
        bars = random.sample(bars, max)

    x, y = [], []
    for a0, a1, v in bars:
        x += [a0, a1, np.nan]
        y += [v, v, np.nan]

    if style == 'book':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=4, color=color, label=label)
    elif style == 'talk':
        plt.plot(x, y, 's-', mew=1, mec='w', ms=0, alpha=1.0, color=colors[2], linewidth=15, label=label)
    else:
        raise ValueError(f'Unrecognized style: {style}')



################################################################################
#########################   MAIN FUNCTIONS   ###################################
################################################################################


def initiliaze_pipeline(filepath, verbose=False):
    ############## 1. Load inputs data ##########################################
    input_data      = pd.read_csv(f'{filepath}/input_data.csv')
    output_template = pd.read_csv(f'{filepath}/output_template.csv')
    parameters      = load_any(f'{filepath}/parameters.jsonc')    
    hierarchy       = load_any(f'{filepath}/hierarchy.json')       
    nodes_to_fit    = load_any(f'{filepath}/nodes_to_fit.json') 

    # create region_id_graph with hierarchy
    nodes = hierarchy['nodes']
    name_to_id = {} # NOTE: this can't handle duplicate names
    id_to_name = {}

    region_id_graph = nx.DiGraph()
    for node in nodes:
        name_to_id[node[0]] = node[1]['location_id']
        id_to_name[node[1]['location_id']] = node[0]

        # add nodes with location_id as the key
        region_id_graph.add_node(
                                node[1]['location_id'],           # location_id is the node key
                                level = node[1]['level'],
                                parent_id = node[1]['parent_id'],
                                name = node[0]
                                )

        # add edges between nodes (ignore root node)
        my_id = node[1]['location_id']
        parent_id = node[1]['parent_id']
        if my_id != parent_id: # ignores root node
            region_id_graph.add_edge(parent_id, my_id)

    # since the graph is a tree, the number of nodes should be equal to the number of edges + 1
    assert region_id_graph.number_of_nodes() == region_id_graph.number_of_edges() + 1, \
        "number of nodes should be equal to the number of edges + 1"
    
    ############## 2. Initialize pm.Model() and shared_data #####################
    pm_model = pm.Model()
    pm_model.shared_data = {     # NOTE: this is what used to be "vars" from class ModelVars
        "input_data"             : input_data,
        "output_template"        : output_template,
        "region_id_graph"        : region_id_graph,
        "id_to_name"             : id_to_name,
        "name_to_id"             : name_to_id,
        "parameters"             : parameters,
        "nodes_to_fit"           : nodes_to_fit,
    }

    if verbose:
        print(f'number of rows: {len(input_data)}')
        print(f'number of unique location_id: {input_data["location_id"].nunique()}')
        print(f"number of nodes: {region_id_graph.number_of_nodes()}") 
        print(f"number of edges: {region_id_graph.number_of_edges()}")
        
    return pm_model



def generate_pymc_objects(
        pm_model, 
        data_type            = 'p',
        lower_bound          = None,
        interpolation_method = 'linear',
        include_covariates   = True,
        mu_age               = None,
        mu_age_parent        = None,
        sigma_age_parent     = None,
        reference_area       = 'Global',
        reference_sex        = 'Both',
        reference_year       = 'all',
        rate_type            = 'neg_binom',
        zero_re              = False
    ):

    ############# 1. Store Parameters to shared_data #########################################################
    pm_model.shared_data['data_type']            = data_type
    pm_model.shared_data['interpolation_method'] = interpolation_method
    pm_model.shared_data['mu_age']               = mu_age
    pm_model.shared_data['mu_age_parent']        = mu_age_parent
    pm_model.shared_data['sigma_age_parent']     = sigma_age_parent
    pm_model.shared_data['reference_area_id']    = pm_model.shared_data['name_to_id'][reference_area]
    pm_model.shared_data['reference_sex']        = reference_sex
    pm_model.shared_data['reference_year']       = reference_year
    pm_model.shared_data['rate_type']            = rate_type
    pm_model.shared_data['zero_re']              = zero_re

    ############# 2. Filter input_data and parameters by data_type (optional: lower_bound) #####################
    input_data          = pm_model.shared_data['input_data']
    data                = input_data[input_data['data_type'] == data_type]
    lb_data             = input_data[input_data['data_type'] == lower_bound] if lower_bound else None
    params_of_data_type = pm_model.shared_data['parameters'][data_type]    
    
    pm_model.shared_data['data']                = data
    pm_model.shared_data['lower_bound']         = lb_data
    pm_model.shared_data['params_of_data_type'] = params_of_data_type

    ############# 3. Fetch ages and age_weights from parameters #####################
    parameters   = pm_model.shared_data['parameters']
    ages         = np.array(parameters['ages'], dtype=np.float64)
    ages_weights = np.array(parameters['age_weights'], dtype=np.float64)

    pm_model.shared_data['ages']        = ages
    pm_model.shared_data['age_weights'] = ages_weights

    ############# 4. Generate knots and smoothing for spline.spline #########################################################
    knots = np.array(params_of_data_type.get('parameter_age_mesh', np.arange(ages[0], ages[-1] + 1, 5)), dtype=np.float64)

    smooth_map = {'No Prior': np.inf, 'Slightly': 0.5, 'Moderately': 0.05, 'Very': 0.005}  # TMI: type(np.inf) == float
    smoothness_param = params_of_data_type.get('smoothness')
    if isinstance(smoothness_param, dict): 
        amount = smoothness_param.get('amount')

        if isinstance(amount, (int, float)): # smoothness_param is dict, and amount is int or float
            smoothing = float(amount)
        else:                                # smoothness_param is dict, and amount may be string
            smoothing = smooth_map.get(amount, 0.0)

    else:                                    # smoothness_param may be string
        smoothing = smooth_map.get(smoothness_param, 0.0)
    
    pm_model.shared_data['knots']    = knots         
    pm_model.shared_data['smoothing'] = smoothing # NOTE: smoothing is eventually just a float like 0.5

    ############# 5. Check Standard Deviation and Effective Sample Size for likelihood.* #######################################
    data = data.copy()
    # identify rows where SE is “invalid” (< 0) or missing, and recompute them
    invalid_se_mask = (data['standard_error'] < 0) | (data['standard_error'].isna())
    se_replacement   = (data['upper_ci'] - data['lower_ci']) / (2 * 1.96)
    se               = data['standard_error'].mask(invalid_se_mask, se_replacement)
    num_se_augmented = int(invalid_se_mask.sum())

    # identify rows where ESS is "invalid" (< 0) or missing, and recompute them
    invalid_ess_mask = (data['effective_sample_size'] < 0) | (data['effective_sample_size'].isna())
    ess_replacement  = data['value'] * (1 - data['value']) / se**2
    ess              = data['effective_sample_size'].mask(invalid_ess_mask, ess_replacement)
    num_ess_augmented = int(invalid_ess_mask.sum())

    # write back and report
    data['standard_error'] = se
    data['effective_sample_size'] = ess
    print(f"Standard errors replaced: {num_se_augmented}")
    print(f"Effective sample sizes filled: {num_ess_augmented}")


    ############# I. Generate PYMC objects #########################################################
    with pm_model:
        ############ Calculating constrained_mu_age #########################################################
        if mu_age is not None:
            unconstrained_mu_age = mu_age
        else:
            unconstrained_mu_age = spline.spline()

        constrained_mu_age = priors.level_constraints(unconstrained_mu_age)
        priors.derivative_constraints(mu_age=constrained_mu_age)            
    
        if mu_age_parent is not None: # penalize based on similarity to parent
            priors.similar(
                mu_child         = constrained_mu_age,
                mu_parent        = mu_age_parent,
                sigma_parent     = sigma_age_parent,
                sigma_difference = 0.0,
                eps              = 1e-9,
                penalty_name     = "_mu_age_parent_not_none"
            )

        ############ Calculating Pi #########################################################
        if len(data) > 0:
            mu_interval = age_groups.age_standardize_approx(mu_age=constrained_mu_age)

            # covariate & pi
            if include_covariates:
                pi, U, U_shift, sigma_alpha, alpha, alpha_potentials, const_alpha_sigma, X, X_shift, beta, const_beta_sigma = covariates.mean_covariate_model(mu=mu_interval)

            else:
                pi = mu_interval

        if len(data) <= 0:
            if include_covariates:
                pi, U, U_shift, sigma_alpha, alpha, alpha_potentials, const_alpha_sigma, X, X_shift, beta, const_beta_sigma = covariates.mean_covariate_model(mu=None)
            else:
                assert False, "shouldn't be here"

        ############ Likelihood based on rate_type #########################################################
        if len(data) > 0:
            if rate_type == 'poisson':
                likelihood.poisson(pi=pi)

            elif rate_type == 'normal':
                sigma = pm.Uniform(
                    name=f'sigma_{data_type}',
                    lower=1e-4,
                    upper=1e-1,
                    initval=1e-2
                )
                likelihood.normal(pi=pi, sigma=sigma)

            elif rate_type == 'log_normal':
                sigma = pm.Uniform(
                    name=f'sigma_{data_type}',
                    lower=1e-4,
                    upper=1.0,
                    initval=1e-2
                )
                likelihood.log_normal(pi=pi, sigma=sigma)

            elif rate_type == 'offset_log_normal':
                sigma= pm.Uniform(
                    name=f'sigma_{data_type}',
                    lower=1e-4,
                    upper=10.0,
                    initval=1e-2
                )
                likelihood.offset_log_normal(pi=pi, sigma=sigma)

            elif rate_type == 'binom':
                likelihood.binom(pi=pi)

            elif rate_type == 'neg_binom':
                hetero = parameters.get('heterogeneity', None)
                lower = {'Slightly': 9.0, 'Moderately': 3.0, 'Very': 1.0}.get(hetero, 1.0)
                if data_type == 'pf':
                    lower = 1e12
                delta = covariates.dispersion_covariate_model(delta_lb=lower, delta_ub=lower * 9.0)
                likelihood.neg_binom(pi=pi, delta=delta)     

            elif rate_type == 'beta_binom':
                hetero = parameters.get('heterogeneity', None)
                lower = {'Slightly': 9.0, 'Moderately': 3.0, 'Very': 1.0}.get(hetero, 1.0)
                if data_type == 'pf':
                    lower = 1e12
                delta = covariates.dispersion_covariate_model(delta_lb=lower, delta_ub=lower * 9.0)
                likelihood.beta_binom(pi=pi, delta=delta)

            else:
                raise ValueError(f'Unsupported rate_type "{rate_type}"')
            
        ############ Covariate Level Constraints #########################################################
        if include_covariates:
            priors.covariate_level_constraints(X_shift, beta, U, alpha, constrained_mu_age)

        ############ Lower Bound #########################################################################
        if lb_data is not None and len(lb_data) > 0:
            lb = {}
            mu_interval_lb = age_groups.age_standardize_approx(mu_age=constrained_mu_age, use_lb_data=True)

            if include_covariates:
                pi_lb, _, _, _, _, _, _, _, _, _, _ = covariates.mean_covariate_model(mu=mu_interval_lb, use_lb_data=True)
            else:
                pi_lb = mu_interval_lb

            delta_lb = covariates.dispersion_covariate_model(lower=1e12, upper=1e13, use_lb_data=True)

            se_lb = lb_data['standard_error'].mask(
                lb_data['standard_error'].le(0) | lb_data['standard_error'].isna(),
                (lb_data['upper_ci'] - lb_data['lower_ci']) / (2 * 1.96)
            )
            ess_lb = lb_data['effective_sample_size'].fillna(
                lb_data['value'] * (1 - lb_data['value']) / se_lb**2
            )
            lb_data['standard_error'] = se_lb
            lb_data['effective_sample_size'] = ess_lb

            likelihood.neg_binom_lower_bound(pi=pi_lb, delta=delta_lb)


        ############ Store Reuseable Variables for predict_for() #########################################################
        if include_covariates:
            pm_model.shared_data['alpha'] = alpha
            pm_model.shared_data['const_alpha_sigma'] = const_alpha_sigma
            pm_model.shared_data['beta'] = beta
            pm_model.shared_data['const_beta_sigma'] = const_beta_sigma
            pm_model.shared_data['X'] = X
            pm_model.shared_data['X_shift'] = X_shift
            pm_model.shared_data['U'] = U
            pm_model.shared_data['U_shift'] = U_shift




def return_map_estimate(pm_model, verbose=False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    verbose = True

    with pm_model:
        if verbose:
            logger.info("  ▶ pm.find_MAP() 수행 중...")
        
        map_estimate = pm.find_MAP()
    return map_estimate



def return_idata(
    pm_model, 
    map_estimate,
    draws         = 2000,
    tune          = 1000,
    chains        = 4,
    cores         = 4,
    target_accept = 0.9,
    max_treedepth = 10,
    use_advi = False,
    use_metropolis = True,
    vi_iters = 20000,
    vi_lr = 1e-3,
    verbose = False,
    ):

    t_start = time.time()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    with pm_model:
        if use_advi:
            if verbose:
                logger.info("  ▶ ADVI 수행 중...")
            approx = pm.fit(
                n=vi_iters,
                method="advi",
                obj_optimizer=pm.adam(learning_rate=vi_lr),
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
            )
            idata = approx.sample(draws=draws)

        elif use_metropolis:
            if verbose:
                logger.info("  ▶ Metropolis 샘플링 수행 중...")
            step = pm.Metropolis()
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step,
                initvals=map_estimate,
                return_inferencedata=True,
                progressbar=verbose,
            )

        else:
            if verbose:
                logger.info("  ▶ NUTS 샘플링 수행 중...")

            print("advi warm up")
            advi = pm.fit(method="advi", n=5000)
                
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                initvals=map_estimate,
                target_accept=target_accept,
                nuts={"max_treedepth": max_treedepth},
                return_inferencedata=True,
                progressbar=verbose,
            )
        
    t_end = time.time()
    wall_time = t_end - t_start
    if verbose:
        logger.info(f"[asr] 전체 소요 시간: {wall_time:.1f}초")

    return idata



def predict_for(
    pm_model,
    idata, 
    root_area           = 'Global',
    root_sex            = 'Both',
    root_year           = 2009,
    area                = 'Global',
    sex                 = 'Female',
    year                = 2005,
    population_weighted = 1.0,
    lower               = 0.0,
    upper               = 1.0,
    include_covariates  = True,
    ):
    ############ Fetch Reuseable Variables #########################################################
    if include_covariates:
        alpha = pm_model.shared_data['alpha']
        const_alpha_sigma = pm_model.shared_data['const_alpha_sigma']
        beta = pm_model.shared_data['beta']
        const_beta_sigma = pm_model.shared_data['const_beta_sigma']
        X = pm_model.shared_data['X']
        X_shift = pm_model.shared_data['X_shift']
        output_template = pm_model.shared_data['output_template']
        region_id_graph = pm_model.shared_data['region_id_graph']
        name_to_id = pm_model.shared_data['name_to_id']
        U = pm_model.shared_data['U']
        U_shift = pm_model.shared_data['U_shift']
        id_to_name = pm_model.shared_data['id_to_name']


    arr = idata.posterior['constrained_mu_age_p'].values
    n_chain, n_draw, n_ages = arr.shape                 # (4, 2000, 93)
    mu_trace = arr.reshape((n_chain * n_draw, n_ages))  # shape = (n_samples, n_ages) -> (8000, 93)
    n_samples = mu_trace.shape[0]

    if not include_covariates:
        return np.clip(mu_trace, lower, upper)

    # Alpha_trace (random effects)
    alpha_trace = np.empty((n_samples, 0))
    if isinstance(alpha, list) and alpha:
        traces = []
        for alpha_node, sigma_const in zip(alpha, const_alpha_sigma):
            name_alpha = alpha_node.name
            # print(name_alpha) 
            if name_alpha in idata.posterior:
                arr_a = idata.posterior[name_alpha].values  # (chains, draws)
                traces.append(arr_a.reshape(n_chain * n_draw))
            else:
                sig = max(sigma_const, 1e-9)
                loc = float(alpha_node)
                draws = np.random.normal(loc=loc, scale=1.0 / np.sqrt(sig), size=n_samples)
                traces.append(draws)
        alpha_trace = np.column_stack(traces)

    
    # 4) beta_trace (fixed effects) 생성
    beta_trace = np.empty((n_samples, 0))
    if isinstance(beta, list) and beta:
        traces = []
        for beta_node, sigma_const in zip(beta, const_beta_sigma):
            name_beta = beta_node.name
            # print(name_beta)
            if name_beta in idata.posterior:
                arr_b = idata.posterior[name_beta].values  # (chains, draws)
                traces.append(arr_b.reshape(n_chain * n_draw))
            else:
                sig = max(sigma_const, 1e-9)
                loc = float(beta_node)
                draws = np.random.normal(loc=loc, scale=1.0 / np.sqrt(sig), size=n_samples)
                traces.append(draws)
        beta_trace = np.column_stack(traces)


    # 5) leaf-nodes 찾기
    leaves = [n for n in nx.bfs_tree(region_id_graph, name_to_id[area]) if region_id_graph.out_degree(n) == 0]
    if not leaves:
        leaves = [name_to_id[area]]

    # 6) output_template에서 (area, sex, year)에 해당하는 pop, covariates 추출
    output_tpl = output_template.copy()
    grp = (
        output_tpl
        .groupby(["area", "sex", "year"], as_index=False)
        .mean()
        .set_index(["area", "sex", "year"])
    )
    # len(grp) is equal to lins in output_template.csv


    SEX_VALUE = {'Male': .5, 'Both': 0., 'Female': -.5}
    # 7) X_df (centered covariates) 준비
    if isinstance(X, pd.DataFrame) and not X.empty:
        # (1) 원래 vars["X"].columns에 들어있는 이름들로 grp에서 필터
        X_df = grp.filter(X.columns, axis=1).copy()

        # (2) "x_sex"가 vars["X"].columns에 있으면 강제로 생성
        if "x_sex" in X.columns:
            X_df["x_sex"] = SEX_VALUE[sex]

        # (3) shift(centering) 적용
        X_df = X_df - X_shift

    else:
        X_df = pd.DataFrame(index=grp.index)

    
    # 8) U_row Series 준비 (한 행짜리)
    if isinstance(U, pd.DataFrame) and not U.empty:
        U_cols = U.columns
        U_row = pd.Series(0.0, index=U_cols)
    else:
        U_row = pd.Series(dtype=float)


    # 9) 각 leaf별로 cov_shift 계산
    cov_shift = np.zeros(n_samples)
    total_weight = 0.0

    for leaf in leaves:
        # (1) U_row 재설정
        U_row[:] = 0.0
        path = nx.shortest_path(region_id_graph, name_to_id[root_area], leaf)
        for node in path[1:]:
            if node in U_row.index:
                U_row[node] = 1.0 - U_shift.get(node, 0.0)

        # (2) random-effect 기여: alpha_trace · U_row
        if alpha_trace.size > 0:
            log_shift = alpha_trace.dot(U_row.values)
        else:
            log_shift = np.zeros(n_samples)

        # (3) fixed-effect 기여: beta_trace · X_vals
        if beta_trace.size and (leaf, sex, year) in X_df.index:
            x_vals = X_df.loc[(leaf, sex, year)].values
            log_shift = log_shift + beta_trace.dot(x_vals)

        # (4) population‐weight or unweighted average
        pop = float(grp.at[(id_to_name[leaf], sex, year), "pop"])

        if population_weighted:
            cov_shift += np.exp(log_shift) * pop
            total_weight += pop
        else:
            cov_shift += log_shift
            total_weight += 1.0


    # (5) 정규화
    if population_weighted:
        cov_shift = cov_shift / total_weight
    else:
        cov_shift = np.exp(cov_shift / total_weight)

    # 10) baseline mu_age와 곱하고 clip
    preds = mu_trace * cov_shift[:, None]  # shape = (n_samples, n_ages)
    return np.clip(preds, lower, upper)


def visualize_pred(pred, data):
    plt.figure(figsize=(10, 4))
    data_bars(
        df=data,
        color='grey',
        label='Simulated PD Data'
    )

    hpd = pm.stats.hdi(pred, hdi_prob=0.95)
    ages = np.arange(pred.shape[1])

    plt.plot(
        ages,
        pred.mean(axis=0),
        'k-', linewidth=2,
        label='Posterior Mean'
    )
    plt.plot(
        ages,
        hpd[:, 0],
        'k--', linewidth=1,
        label='95% HPD interval'
    )
    plt.plot(
        ages,
        hpd[:, 1],
        'k--', linewidth=1
    )

    plt.xlabel('Age (years)')
    plt.ylabel('Prevalence (per 1)')
    plt.grid()
    plt.legend(loc='upper left')
    plt.axis(ymin=-0.001, xmin=-5, xmax=105)
