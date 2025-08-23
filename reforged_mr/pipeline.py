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
import matplotlib.pyplot as plt

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


def initialize_pipeline(input_data_path, output_template_path, parameters_path, hierarchy_path, detailed_pop_path, verbose=False):
    ############## 1. Load inputs data ##########################################
    input_data      = pd.read_csv(input_data_path)
    output_template = pd.read_csv(output_template_path)
    detailed_pop    = pd.read_csv(detailed_pop_path)
    parameters      = load_any(parameters_path)    
    hierarchy       = load_any(hierarchy_path)       
    # nodes_to_fit    = load_any(f'{filepath}/nodes_to_fit.json') 

    # create region_id_graph with hierarchy
    nodes = hierarchy['nodes']
    name_to_id = {} # NOTE: this can't handle duplicate names
    id_to_name = {}

    region_id_graph = nx.DiGraph()
    for node in nodes:
        node_id = int(node[0])
        node_name = node[1]['location_name']
        node_level = int(node[1]['level'])
        node_parent_id = int(node[1]['parent_id'])

        name_to_id[node_name] = node_id
        id_to_name[node_id] = node_name

        # add nodes with location_id as the key
        region_id_graph.add_node(
                                node_id,           # location_id is the node key
                                level = node_level,
                                parent_id = node_parent_id,
                                name = node_name
                                )

        # add edges between nodes (ignore root node)
        if node_id != node_parent_id: # ignores root node
            region_id_graph.add_edge(node_parent_id, node_id)
    
    assert nx.is_tree(region_id_graph), "region_id_graph is not a tree"

    # # since the graph is a tree, the number of nodes should be equal to the number of edges + 1
    # assert region_id_graph.number_of_nodes() == region_id_graph.number_of_edges() + 1, \
    #     "number of nodes should be equal to the number of edges + 1"
    
    ############## 2. Initialize pm.Model() and shared_data #####################
    pm_model = pm.Model()
    pm_model.shared_data = {     # NOTE: this is what used to be "vars" from class ModelVars
        "input_data"             : input_data,
        "output_template"        : output_template,
        "region_id_graph"        : region_id_graph,
        "id_to_name"             : id_to_name,
        "name_to_id"             : name_to_id,
        "parameters"             : parameters,
        "detailed_pop"           : detailed_pop,
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
        zero_re              = True
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
    params_of_data_type = pm_model.shared_data['parameters'][data_type]    
    
    pm_model.shared_data['data']                = data
    pm_model.shared_data['params_of_data_type'] = params_of_data_type

    ############# 3. Fetch ages and age_weights from parameters #####################
    parameters   = pm_model.shared_data['parameters']
    ages         = np.array(parameters['ages'], dtype=np.float64)
    ages_weights = np.array(parameters['age_weights'], dtype=np.float64)

    pm_model.shared_data['ages']        = ages
    pm_model.shared_data['age_weights'] = ages_weights

    ############# 4. Generate knots and smoothing for spline.spline #########################################################
    knots = np.array(params_of_data_type.get('parameter_age_mesh', np.arange(ages[0], ages[-1] + 1, 5)), dtype=np.float64)
    if knots[-1] != ages[-1]:
        knots = np.concatenate([knots, [ages[-1]]])
    pm_model.shared_data['knots']    = knots 
    
    smooth_map = {'No Prior': np.inf, 'Slightly': 0.5, 'Moderately': 0.05, 'Very': 0.005}  # TMI: type(np.inf) == float

    # params_of_data_type 에서 가져온 후
    smoothness_param = params_of_data_type.get('smoothness')

    if not isinstance(smoothness_param, dict):
        raise ValueError(
            "‘smoothness’ must be a dict with keys "
            "{'age_start', 'amount', 'age_end'}"
        )

    required_keys = {'age_start', 'amount', 'age_end'}
    if set(smoothness_param.keys()) != required_keys:
        raise ValueError(
            "‘smoothness’ dict must contain exactly the keys "
            f"{required_keys}, but got {set(smoothness_param.keys())}"
        )

    amount = smoothness_param['amount']

    if isinstance(amount, (int, float)):
        smoothing = float(amount)

    elif isinstance(amount, str):
        if amount not in smooth_map:
            raise ValueError(
                f"Invalid smoothness amount '{amount}'. "
                f"Expected one of {list(smooth_map.keys())}."
            )
        smoothing = smooth_map[amount]

    else:
        raise TypeError(
            f"‘amount’ must be int, float, or one of {list(smooth_map.keys())}, "
            f"got {type(amount).__name__}"
        )
        
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

    pm_model.shared_data['data'] = data


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
                pi, U, U_ref, sigma_alpha, alpha, alpha_potentials, const_alpha_sigma, X, X_centering, X_scaling, beta, const_beta_sigma = covariates.mean_covariate_model(mu=mu_interval)

            else:
                pi = mu_interval

        if len(data) <= 0:
            if include_covariates:
                pi, U, U_ref, sigma_alpha, alpha, alpha_potentials, const_alpha_sigma, X, X_centering, X_scaling, beta, const_beta_sigma = covariates.mean_covariate_model(mu=None)
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
            priors.covariate_level_constraints(X_centering, X_scaling, beta, U, alpha, constrained_mu_age)

        ############ Store Reuseable Variables for predict_for() #########################################################
        if include_covariates:
            pm_model.shared_data['alpha'] = alpha
            pm_model.shared_data['const_alpha_sigma'] = const_alpha_sigma
            pm_model.shared_data['beta'] = beta
            pm_model.shared_data['const_beta_sigma'] = const_beta_sigma
            pm_model.shared_data['X'] = X
            pm_model.shared_data['X_centering'] = X_centering
            pm_model.shared_data['X_scaling'] = X_scaling
            pm_model.shared_data['U'] = U
            pm_model.shared_data['U_ref'] = U_ref




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


def visualize_pred(pred, data, save_path=None):
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

    # ---------- 저장 옵션 ----------
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figure saved to {save_path}")

    plt.show()


def predict_for(
    pm_model,
    idata, 
    root_area           = 'Global',
    root_sex            = 'Both',
    root_year           = 'all',
    location_id         = 1,
    sex_name            = 'Both',
    year_id             = 2005,
    population_weighted = True,
    lower               = 0.0,
    upper               = 1.0,
    include_covariates  = True,
    return_scalar       = True,
):
    """
    반환:
      - return_scalar == True  → dict {
            'prevalence': (n_samples,)  연령가중 평균 유병률
            'cases'     : (n_samples,)  절대 환자 수 (유병률×인구)
        }
      - return_scalar == False → (n_samples, n_ages) : 연령별 곡선
    가중치는 항상 detailed_pop(성별 포함)을 사용. 요청 성별 인구가 없으면 ValueError.
    """

    # -------------------- 0) baseline mu_age (draw x age) --------------------
    arr = idata.posterior['constrained_mu_age_p'].values
    n_chain, n_draw, n_ages = arr.shape
    mu_trace = arr.reshape((n_chain*n_draw, n_ages))

    ages = np.asarray(pm_model.shared_data['ages'], dtype=float)
    assert len(ages) == n_ages, f"Age axis mismatch: len(ages)={len(ages)} vs n_ages={n_ages}"

    n_samples = mu_trace.shape[0]
    age_index = np.arange(n_ages)

    # 항상 필요한 공용 데이터
    region_id_graph = pm_model.shared_data['region_id_graph']
    detailed_pop    = pm_model.shared_data['detailed_pop']

    # -------------------- 1) 공변량/RE 미포함 모드 --------------------
    if not include_covariates:
        if den == 0 or not np.isfinite(den):
            raise ValueError("den is zero or non-finite.")
        # leaf(국가) 수집
        if location_id in region_id_graph:
            leaf_ids = [n for n in nx.bfs_tree(region_id_graph, location_id)
                        if region_id_graph.out_degree(n) == 0]
            if not leaf_ids:
                leaf_ids = [location_id]
        else:
            leaf_ids = [location_id]

        if return_scalar:
            num_prev  = np.zeros(n_samples)
            num_cases = np.zeros(n_samples)
            den = 0.0

            for leaf in leaf_ids:
                # ★ 성별 포함 strict 가중치
                w = _pop_weights_for_leaf(detailed_pop, leaf, year_id, age_index, sex_name)
                ws = w.sum()
                if ws <= 0:
                    continue
                num_prev  += (mu_trace * w[None, :]).sum(axis=1)
                num_cases += (mu_trace * w[None, :]).sum(axis=1)
                den += ws

            if den <= 0:
                raise ValueError(f"[predict_for] detailed_pop empty: loc={location_id}, year={year_id}, sex={sex_name}")

            prevalence = np.clip(num_prev / den, lower, upper)
            cases      = num_cases  # (유병률 × 인구)의 합
            return {"prevalence": prevalence, "cases": cases}

        # 곡선 반환 (공변량/RE 없으면 지역/성별과 무관)
        return np.clip(mu_trace, lower, upper)

    # -------------------- 2) 공변량/RE 포함 모드 --------------------
    alpha             = pm_model.shared_data['alpha']
    const_alpha_sigma = pm_model.shared_data['const_alpha_sigma']
    beta              = pm_model.shared_data['beta']
    const_beta_sigma  = pm_model.shared_data['const_beta_sigma']
    X                 = pm_model.shared_data['X']
    X_centering       = pm_model.shared_data['X_centering']
    X_scaling         = pm_model.shared_data['X_scaling']
    output_template   = pm_model.shared_data['output_template']
    U                 = pm_model.shared_data['U']
    U_ref             = pm_model.shared_data['U_ref']

    # alpha_trace (RE)
    alpha_trace = np.empty((n_samples, 0))
    if isinstance(alpha, list) and alpha:
        traces = []
        for alpha_node, sigma_const in zip(alpha, const_alpha_sigma):
            name_alpha = alpha_node.name
            if name_alpha in idata.posterior:
                arr_a = idata.posterior[name_alpha].values  # (C, S)
                traces.append(arr_a.reshape(n_chain * n_draw))
            else:
                sig = _safe_sigma(sigma_const)
                loc = float(alpha_node)
                draws = np.random.normal(loc=loc, scale=1.0/np.sqrt(sig), size=n_samples)
                traces.append(draws)
        alpha_trace = np.column_stack(traces)

    # beta_trace (FE)
    beta_trace = np.empty((n_samples, 0))
    if isinstance(beta, list) and beta:
        traces = []
        for beta_node, sigma_const in zip(beta, const_beta_sigma):
            name_beta = beta_node.name
            if name_beta in idata.posterior:
                arr_b = idata.posterior[name_beta].values  # (C, S)
                traces.append(arr_b.reshape(n_chain * n_draw))
            else:
                sig = _safe_sigma(sigma_const)
                loc = float(beta_node)
                draws = np.random.normal(loc=loc, scale=1.0/np.sqrt(sig), size=n_samples)
                traces.append(draws)
        beta_trace = np.column_stack(traces)

    # leaf nodes
    if location_id in region_id_graph:
        leaf_ids = [n for n in nx.bfs_tree(region_id_graph, location_id)
                    if region_id_graph.out_degree(n) == 0]
        if not leaf_ids:
            leaf_ids = [location_id]
    else:
        leaf_ids = [location_id]

    # X_df 준비
    output_tpl = output_template.copy()
    output_tpl["location_id"] = output_tpl["location_id"].astype(int)
    output_tpl["sex_name"]    = output_tpl["sex_name"].astype(str)
    output_tpl["year_id"]     = output_tpl["year_id"].astype(int)
    grp = output_tpl.set_index(["location_id","sex_name","year_id"]).sort_index()

    SEX_VALUE = {'Male': .5, 'Both': 0., 'Female': -.5}
    if isinstance(X, pd.DataFrame) and not X.empty:
        X_df = grp.filter(X.columns, axis=1).copy()
        if "x_sex" in X.columns:
            X_df["x_sex"] = SEX_VALUE[sex_name]
        X_df = (X_df - X_centering) / X_scaling
    else:
        X_df = pd.DataFrame(index=grp.index)

    # U_row 초기화
    if isinstance(U, pd.DataFrame) and not U.empty:
        U_cols = U.columns
        U_row_template = pd.Series(0.0, index=U_cols)
    else:
        U_row_template = pd.Series(dtype=float)

    # aggregation
    if return_scalar:
        num_prev  = np.zeros(n_samples)
        num_cases = np.zeros(n_samples)
        den = 0.0
    else:
        num = np.zeros((n_samples, n_ages))
        den = np.zeros(n_ages) if population_weighted else 0.0

    leaf_count = 0

    for leaf in leaf_ids:
        # (a) U_row
        if not U_row_template.empty and (leaf in region_id_graph):
            U_row = U_row_template.copy()
            path = nx.shortest_path(region_id_graph, pm_model.shared_data['name_to_id']['Global'], leaf)
            for node in path[1:]:
                if node in U_row.index:
                    U_row[node] = 1.0 - U_ref.get(node, 0.0)
        else:
            U_row = pd.Series(dtype=float)

        # (b) log_shift = alpha·U + beta·x
        if alpha_trace.size > 0 and not U_row.empty:
            log_shift = alpha_trace.dot(U_row.values)
        else:
            log_shift = np.zeros(n_samples)

        if beta_trace.size > 0 and ((leaf, sex_name, year_id) in X_df.index):
            x_vals = X_df.loc[(leaf, sex_name, year_id)].values
            log_shift = log_shift + beta_trace.dot(x_vals)

        # (c) 예측 곡선
        preds_leaf = mu_trace * np.exp(log_shift)[:, None]
        preds_leaf = np.clip(preds_leaf, lower, upper)

        # (d) 성별 포함 strict 가중치
        w = _pop_weights_for_leaf(detailed_pop, leaf, year_id, age_index, sex_name)
        ws = w.sum()
        if ws <= 0:
            continue

        if return_scalar:
            num_prev  += (preds_leaf * w[None, :]).sum(axis=1)
            num_cases += (preds_leaf * w[None, :]).sum(axis=1)
            den += ws
        else:
            if population_weighted:
                num += preds_leaf * w[None, :]
                den += w
            else:
                num += preds_leaf
                leaf_count += 1

    # finalize
    if return_scalar:
        if den <= 0:
            raise ValueError(f"[predict_for] no population: loc={location_id}, year={year_id}, sex={sex_name}")
        prevalence = np.clip(num_prev / den, lower, upper)
        cases      = num_cases
        return {"prevalence": prevalence, "cases": cases}
    else:
        if population_weighted:
            den_safe = np.where(den > 0, den, 1e-12)
            preds_curve = num / den_safe[None, :]
            return np.clip(preds_curve, lower, upper)
        else:
            if leaf_count == 0:
                raise ValueError("no valid leaf")
            preds_curve = num / leaf_count
            return np.clip(preds_curve, lower, upper)

# ------------ 헬퍼: NaN/비정상 sigma 방어 ------------
def _safe_sigma(sigma_const):
    try:
        sig = float(sigma_const)
        if not np.isfinite(sig) or sig <= 0:
            return 1.0
        return sig
    except Exception:
        return 1.0

# ------------ 헬퍼: 성별 포함 strict 인구 가중치 ------------
def _pop_weights_for_leaf(dpop, leaf_id, year_id, ages, sex_name):
    ages = np.asarray(ages, dtype=float)

    df = dpop.copy()
    df['location_id'] = df['location_id'].astype(int)
    df['year_id']     = df['year_id'].astype(int)
    df['sex_name']    = df['sex_name'].astype(str).str.strip()
    df['age']         = df['age'].astype(float)

    sel = df[(df['location_id']==int(leaf_id)) &
             (df['year_id']==int(year_id)) &
             (df['sex_name']==str(sex_name).strip())]

    if sel.empty:
        raise ValueError(f"[pop_weights] missing: loc={leaf_id}, year={year_id}, sex={sex_name}")

    # 중복 나이 집계 후 정확 라벨로 reindex
    sel = sel.groupby('age', as_index=False, sort=False)['value'].sum()
    s = sel.set_index('age')['value']
    idx = pd.Index(ages)
    w = s.reindex(idx).fillna(0.0).to_numpy(float)

    # 커버리지 경고/에러
    covered = (w > 0).sum()
    if covered == 0:
        pop_min, pop_max = float(sel['age'].min()), float(sel['age'].max())
        raise ValueError(
            f"[pop_weights] no overlapping ages: model [{ages.min()}..{ages.max()}], "
            f"pop [{pop_min}..{pop_max}] for loc={leaf_id}, year={year_id}, sex={sex_name}"
        )
    return w




def world_predict(
    pm_model,
    idata,
    years,              
    output_csv_path
):
    # 1) 그래프/이름 매핑
    region_id_graph = pm_model.shared_data['region_id_graph']
    id_to_name      = pm_model.shared_data['id_to_name']
    ages            = pm_model.shared_data['ages']
    age_weights_in  = pm_model.shared_data['age_weights']

    ages = np.asarray(ages, dtype=int)
    age_w = _as_age_weight_vector(age_weights_in, ages)

    # 2) level 0,2,3 노드만
    target_levels = {0, 2, 3}
    nodes = []
    for nid_str, data in region_id_graph.nodes(data=True):
        level = data.get('level', None)
        if level in target_levels:
            try:
                nid_int = int(nid_str)
            except (TypeError, ValueError):
                continue
            nodes.append((nid_int, level, id_to_name.get(nid_int, str(nid_int))))

    # ✅ GBD 관행 sex_id 매핑
    sex_id_map = {'Male': 1, 'Female': 2, 'Both': 3}

    rows = []
    for year in years:
        for sex in ['Both', 'Male', 'Female']:
            for loc_id, level, loc_name in nodes:
                try:
                    # 1) 스칼라(유병률/환자수)
                    res_scalar = predict_for(
                        pm_model,
                        idata,
                        root_area='Global',
                        root_sex='Both',
                        root_year='all',
                        location_id=loc_id,
                        sex_name=sex,
                        year_id=int(year),
                        population_weighted=True,
                        lower=0.0,
                        upper=1.0,
                        include_covariates=True,
                        return_scalar=True,
                    )
                    prev_samples  = res_scalar["prevalence"]
                    cases_samples = res_scalar["cases"]

                    # 2) 연령표준화 유병률 (곡선 한 번 더)
                    preds_curve = predict_for(
                        pm_model,
                        idata,
                        root_area='Global',
                        root_sex='Both',
                        root_year='all',
                        location_id=loc_id,
                        sex_name=sex,
                        year_id=int(year),
                        population_weighted=True,
                        lower=0.0,
                        upper=1.0,
                        include_covariates=True,
                        return_scalar=False,
                    )
                    if preds_curve.ndim != 2 or preds_curve.shape[1] != len(age_w):
                        raise ValueError(
                            f"preds_curve shape {preds_curve.shape} != age_weights length {len(age_w)}"
                        )
                    prev_std_samples = preds_curve @ age_w  # (n_samples,)

                    # 요약
                    mean_prev     = float(np.mean(prev_samples))
                    lower_prev    = float(np.percentile(prev_samples, 2.5))
                    upper_prev    = float(np.percentile(prev_samples, 97.5))

                    mean_cases    = float(np.mean(cases_samples))
                    lower_cases   = float(np.percentile(cases_samples, 2.5))
                    upper_cases   = float(np.percentile(cases_samples, 97.5))

                    mean_prev_std   = float(np.mean(prev_std_samples))
                    lower_prev_std  = float(np.percentile(prev_std_samples, 2.5))
                    upper_prev_std  = float(np.percentile(prev_std_samples, 97.5))

                    rows.append({
                        "location_id":     loc_id,
                        "location_name":   loc_name,
                        "sex_name":        sex,
                        "sex_id":          sex_id_map[sex],
                        "level":           level,
                        "year":            int(year),
                        "mean_prev":       mean_prev,
                        "lower_prev":      lower_prev,
                        "upper_prev":      upper_prev,
                        "mean_cases":      mean_cases,
                        "lower_cases":     lower_cases,
                        "upper_cases":     upper_cases,
                        "mean_prev_std":   mean_prev_std,
                        "lower_prev_std":  lower_prev_std,
                        "upper_prev_std":  upper_prev_std,
                    })

                except ValueError as e:
                    print(f"[world_predict] Skip loc={loc_id} ({loc_name}), year={year}, sex={sex} :: {e}")
                except Exception as e:
                    print(f"[world_predict] Error  loc={loc_id} ({loc_name}), year={year}, sex={sex} :: {e}")

    # 4) 저장
    if len(rows) == 0:
        print("[world_predict] Warning: no rows computed; CSV not written.")
        return pd.DataFrame(columns=[
            "location_id","location_name","sex_name","sex_id","level","year",
            "mean_prev","lower_prev","upper_prev",
            "mean_cases","lower_cases","upper_cases",
            "mean_prev_std","lower_prev_std","upper_prev_std",
        ])

    df = (
        pd.DataFrame(rows)
        .sort_values(["year","level","location_name","sex_id"])
        .reset_index(drop=True)
    )
    df.to_csv(output_csv_path, index=False)
    print(f"[world_predict] Saved {len(df)} rows to '{output_csv_path}'")

    return df

def _as_age_weight_vector(age_weights, ages):
    """
    age_weights를 모델의 ages 순서에 맞춘 1D numpy 벡터로 변환하고,
    합이 0이 아니면 1로 정규화한다.
    허용 입력: dict( age->w ), pandas.Series(index=age), list/ndarray(ages와 동일 길이)
    """
    if isinstance(age_weights, dict):
        vec = np.array([age_weights.get(int(a), 0.0) for a in ages], dtype=float)
    elif isinstance(age_weights, pd.Series):
        s = age_weights.copy()
        # 인덱스가 age라고 가정, reindex 후 NaN은 0으로
        vec = s.reindex(ages).fillna(0.0).to_numpy(dtype=float)
    else:
        vec = np.asarray(age_weights, dtype=float)
        if vec.ndim != 1:
            raise ValueError("age_weights must be 1D.")
        if len(vec) != len(ages):
            raise ValueError(f"age_weights length ({len(vec)}) != len(ages) ({len(ages)}).")

    total = vec.sum()
    if total > 0 and np.isfinite(total):
        vec = vec / total
    return vec