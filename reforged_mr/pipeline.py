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
        data_type                  = 'p',
        lower_bound                = None,
        interpolation_method       = 'linear',
        include_covariates         = True,
        include_only_sex_covariate = False,
        mu_age                     = None,
        mu_age_parent              = None,
        sigma_age_parent           = None,
        reference_area             = 'Global',
        reference_sex              = 'Both',
        reference_year             = 'all',
        rate_type                  = 'neg_binom',
        zero_re                    = False
    ):

    if include_covariates and include_only_sex_covariate:
        raise ValueError('Cannot set both include_covariates and include_only_sex_covariate to True.')

    ############# 1. Store Parameters to shared_data #########################################################
    pm_model.shared_data['data_type']            = data_type
    pm_model.shared_data['interpolation_method'] = interpolation_method
    pm_model.shared_data['mu_age']               = mu_age
    pm_model.shared_data['mu_age_parent']        = mu_age_parent
    pm_model.shared_data['sigma_age_parent']     = sigma_age_parent
    pm_model.shared_data['reference_area_id']    = pm_model.shared_data['name_to_id'][reference_area]
    pm_model.shared_data['reference_sex']        = reference_sex
    pm_model.shared_data['reference_year']       = reference_year
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
            elif include_only_sex_covariate:
                pi, _, _, _, _, _, _, X, X_shift, beta, const_beta_sigma = covariates.mean_covariate_model_only_sex(mu=mu_interval)
            else:
                pi = mu_interval

        if len(data) <= 0:
            if include_covariates:
                pi, U, U_shift, sigma_alpha, alpha, alpha_potentials, const_alpha_sigma, X, X_shift, beta, const_beta_sigma = covariates.mean_covariate_model(mu=None)
            elif include_only_sex_covariate:
                pi, _, _, _, _, _, _, X, X_shift, beta, const_beta_sigma = covariates.mean_covariate_model_only_sex(mu=None)
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
        elif include_only_sex_covariate:
            priors.covariate_level_constraints_only_sex(X_shift, beta, constrained_mu_age)
        ############ Lower Bound #########################################################################
        if lb_data is not None and len(lb_data) > 0:
            lb = {}
            mu_interval_lb = age_groups.age_standardize_approx(mu_age=constrained_mu_age, use_lb_data=True)

            if include_covariates:
                pi_lb, _, _, _, _, _, _, _, _, _, _ = covariates.mean_covariate_model(mu=mu_interval_lb, use_lb_data=True)
            elif include_only_sex_covariate:
                pi_lb, _, _, _, _, _, _, _, _, _, _ = covariates.mean_covariate_model_only_sex(mu=mu_interval_lb, use_lb_data=True)
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

        elif include_only_sex_covariate:
            pm_model.shared_data['beta'] = beta
            pm_model.shared_data['const_beta_sigma'] = const_beta_sigma
            pm_model.shared_data['X'] = X
            pm_model.shared_data['X_shift'] = X_shift




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
            
            # advi = pm.fit(method="advi", n=5000)
            print("no map estimate")
                
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
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
    include_covariates  = True,
    lower               = 0.0,
    upper               = 1.0,
    year                = 2005,
    location_id         = 0,        # id of the node in the area hierarchy
    sex                 = 'Female', # Male or Female (doesn't support Both yet)
    ):
    

    ##############################################################################################
    #
    # Given an year, country and sex, returns the prediction of the prevalence (for each age: 2-94)
    # Returned shape: (n_samples, n_ages)
    #
    # 1. age -> h(a)
    #
    # 2. country + year                    -> sdi
    #    country + year + sex              -> tobacco
    #    (sdi * beta1) + (tobacco * beta2) -> fixed effects
    #
    # 3. country -> available ancestors among (alpha_global + alpha_super_region + alpha_region + alpha_country) : random effects
    #
    # 4. if country is a region, super region, or global, 
    #    then use the weighted sum of the leaves' prevalnce
    #    weight is the population ratio of the leaves
    #
    ##############################################################################################


    # CASE 1: Do not include covariates: Early return once you get h(a)
    h_a_trace = idata.posterior['constrained_mu_age_p'].values
    n_chain, n_draw, n_ages = h_a_trace.shape                 # (4, 2000, 93)
    h_a_trace = h_a_trace.reshape((n_chain * n_draw, n_ages))  # shape = (n_samples, n_ages) -> (8000, 93)
    h_a_trace_clipped = np.clip(h_a_trace, lower, upper)

    # Early return if not include covariates
    if not include_covariates:    
        return h_a_trace_clipped

    # CASE 2: Include covariates: Get the covariates and calculate the prevalence
    n_samples = h_a_trace.shape[0]

    # Get relevant data from pm_model        
    output_template    = pm_model.shared_data['output_template']
    region_id_graph    = pm_model.shared_data['region_id_graph']


    # CASE 2.1: Get beta_trace (fixed effects) : shape = (n_samples, n_ages) -> (8000, 1)
    beta_sdi_trace = idata.posterior['beta_p_x_sdi'].values.reshape(n_samples, -1) # shape = (8000, 1)
    
    beta_tob_trace = idata.posterior['beta_p_x_tob'].values.reshape(n_samples, -1) # shape = (8000, 1)
    
    beta_sex_trace = idata.posterior['beta_p_x_sex'].values.reshape(n_samples, -1) # shape = (8000, 1)
    
    # Use the year and sex to get x_sdi, x_tob from the df, output_template
    # Return a single df of shape (204, 2)
    x_sdi_tob_df = output_template.loc[
        (output_template['year'] == year) & (output_template['sex'] == sex),
        ['location_id', 'x_sdi', 'x_tob']
    ].reset_index(drop=True)
    
    if sex == 'Male':
        x_sdi_tob_df['x_sex'] = 0.5
    elif sex == 'Female':
        x_sdi_tob_df['x_sex'] = -0.5
    
    # check if location_id is in the df
    if location_id not in x_sdi_tob_df['location_id'].values:
        leaves = [n for n in nx.bfs_tree(region_id_graph, location_id) if region_id_graph.out_degree(n) == 0]
    else:
        leaves = [location_id]

    # Filter x_sdi_tob_df using the leaves
    x_sdi_tob_df = x_sdi_tob_df[x_sdi_tob_df['location_id'].isin(leaves)]
    
    # create a df of shape (n_samples, n_leaves) for x_sdi
    x_sdi_values = x_sdi_tob_df.set_index('location_id').reindex(leaves)['x_sdi'].values
    x_sdi_df = pd.DataFrame(np.tile(x_sdi_values, (n_samples, 1)), columns=leaves) # shape = (8000, n_leaves)
    
    # create a df of shape (n_samples, n_leaves) for x_tob
    x_tob_values = x_sdi_tob_df.set_index('location_id').reindex(leaves)['x_tob'].values
    x_tob_df = pd.DataFrame(np.tile(x_tob_values, (n_samples, 1)), columns=leaves) # shape = (8000, n_leaves)
    
    # create a df of shape (n_samples, n_leaves) for x_sex
    x_sex_values = x_sdi_tob_df.set_index('location_id').reindex(leaves)['x_sex'].values
    x_sex_df = pd.DataFrame(np.tile(x_sex_values, (n_samples, 1)), columns=leaves) # shape = (8000, n_leaves)

    # use the leaves and the output_template to get the population (pop)
    pop_df = output_template.loc[
        (output_template['year'] == year) & (output_template['sex'] == sex),
        ['location_id', 'pop']
    ].reset_index(drop=True)
    
    # filter pop_df using the leaves
    pop_df = pop_df[pop_df['location_id'].isin(leaves)]
    pop_df['pop_ratio'] = pop_df['pop'] / pop_df['pop'].sum()  

    # only keep the pop_ratio and location_id
    pop_df = pop_df.set_index('location_id').reindex(leaves)['pop_ratio'].values
    pop_ratio_df = pd.DataFrame(np.tile(pop_df, (n_samples, 1)), columns=leaves)


    # CASE 2.2: Calculate the random effects    
    random_effect_df = pd.DataFrame(np.zeros((n_samples, len(leaves))), columns=leaves)

    for leaf in leaves:
        random_effect_nodes = list(nx.ancestors(region_id_graph, leaf))
        random_effect_nodes.append(leaf)

        for node in random_effect_nodes:
            alpha_name = f'alpha_p_{node}'
            
            alpha_sum = np.zeros((n_samples, 1))
            if alpha_name in idata.posterior:
                alpha_sum += idata.posterior[alpha_name].values.reshape(n_samples, -1)
                
        random_effect_df[leaf] = alpha_sum

    # CASE 2.3: total effect = fixed effect + random effect
    fixed_effect = beta_sdi_trace * x_sdi_df + \
                    beta_tob_trace * x_tob_df + \
                    beta_sex_trace * x_sex_df
    
    total_effect = np.exp(fixed_effect + random_effect_df)
    weighted_total_effect = total_effect * pop_ratio_df
    weighted_total_effect_sum = weighted_total_effect.sum(axis=1).to_numpy().reshape(-1, 1)
    
    pred = h_a_trace_clipped * weighted_total_effect_sum

    if include_covariates:
        print(f"mean: beta_sdi_trace: {beta_sdi_trace.mean()}")
        print(f"mean: beta_tob_trace: {beta_tob_trace.mean()}")
        print(f"mean: beta_sex_trace: {beta_sex_trace.mean()}")

    return pred


def visualize_pred(pred, data, year, area_name, sex, include_covariates):
    # pred: (n_samples, n_ages)

    plt.figure(figsize=(10, 6))

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

    if include_covariates:
        covs = 'with covariates'
    else:
        covs = 'without covariates'

    plt.title(f'Prevalence of AMD: {area_name} ({sex}) in {year}, {covs}')
    plt.xlabel('Age')
    plt.ylabel('Prevalence')
    plt.grid()
    plt.legend(loc='upper left')
    plt.axis(ymin=-0.001, xmin=-5, xmax=105)
