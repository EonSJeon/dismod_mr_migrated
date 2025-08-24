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
import model.priors as priors
import model.age_groups as age_groups
import model.covariates as covariates
import model.likelihood as likelihood



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


################################################################################
#########################   MAIN FUNCTIONS   ###################################
################################################################################


def initialize_pipeline(
    input_data_path,
    output_template_path,
    parameters_path,
    hierarchy_path,
    detailed_pop_path,
    verbose=False,
):
    # ---------- 1) Load ----------
    input_data      = pd.read_csv(input_data_path)
    output_template = pd.read_csv(output_template_path)
    detailed_pop    = pd.read_csv(detailed_pop_path)
    parameters      = load_any(parameters_path)
    hierarchy       = load_any(hierarchy_path)

    # ---------- 2) Build directed tree (parent -> child), id-only ----------
    nodes = hierarchy["nodes"] 
    G = nx.DiGraph()
    id_to_name: dict[int, str] = {}

    for node in nodes:
        node_id        = int(node[0])
        attrs          = node[1]
        node_name      = attrs["location_name"]
        node_level     = int(attrs["level"])
        node_parent_id = int(attrs["parent_id"])

        G.add_node(
            node_id,
            level=node_level,
            parent_id=node_parent_id,
            name=node_name,
        )
        if node_id != node_parent_id:  # skip self-edge for the root
            G.add_edge(node_parent_id, node_id)

        id_to_name[node_id] = node_name

    # ---------- 3) Validate: directed tree (arborescence) ----------
    roots = [n for n, indeg in G.in_degree() if indeg == 0]
    if len(roots) != 1:
        raise ValueError(f"Hierarchy must have exactly one root (found {roots}).")
    global_id = roots[0]

    if not nx.is_arborescence(G):
        bad_indeg = [n for n, d in G.in_degree()
                     if (n == global_id and d != 0) or (n != global_id and d != 1)]
        reachable = set(nx.descendants(G, global_id)) | {global_id}
        orphans   = [n for n in G.nodes if n not in reachable]
        raise ValueError(
            "region_id_graph is not a directed tree (arborescence). "
            f"violations: indegree_bad={bad_indeg}, orphans={orphans}"
        )
    depth_by_node = nx.single_source_shortest_path_length(G, global_id)
    max_depth = (int(max(depth_by_node.values())) if depth_by_node else 0)+ 1

    # ---------- 4) Create model & attach shared data ----------
    pm_model = pm.Model()
    pm_model.shared_data = {
        "input_data"      : input_data,
        "output_template" : output_template,
        "region_id_graph" : G,
        "id_to_name"      : id_to_name,   
        "parameters"      : parameters,
        "detailed_pop"    : detailed_pop,
        "global_id"    : global_id,
        "max_depth"     : max_depth,
    }

    if verbose:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"#rows(input_data): {len(input_data)}")
        print(f"#unique location_id: {input_data['location_id'].nunique()}")
        print(f"#nodes: {n_nodes}, #edges: {n_edges}, global_id: {global_id} ({id_to_name.get(global_id)})")

    return pm_model

def generate_pymc_objects(
        pm_model, 
        data_type            = 'p',
        mu_age               = None,
        mu_age_parent        = None,
        sigma_age_parent     = None,
        reference_area_id    = None,
        reference_sex        = 'Both',
        reference_year       = 'all',
    ):
    sd = pm_model.shared_data

    # ----------------------------------------------------------------------
    # 1) 참조 축(지역/성/연도) 및 부모 곡선 정보를 shared_data에 저장
    #    - 이후 예측/제약/경로 계산 등에서 공통 참조로 사용
    # ----------------------------------------------------------------------
    if reference_area_id is None:
        reference_area_id = sd['global_id']

    sd['reference_area_id'] = reference_area_id
    sd['reference_sex']     = reference_sex
    sd['reference_year']    = reference_year

    sd[f'mu_age_parent_{data_type}']    = mu_age_parent
    sd[f'sigma_age_parent_{data_type}'] = sigma_age_parent
    
    # ----------------------------------------------------------------------
    # 2) data_type별 파라미터/설정 로드
    #    - parameters.json(c)에서 현재 data_type 블록만 추출
    #    - 옵션 미지정 시 기본값을 사용
    # ----------------------------------------------------------------------
    parameters         = sd['parameters']
    params_of_data_type = parameters[data_type]

    include_covariates = params_of_data_type.get('include_covariates', True)
    rate_type          = params_of_data_type.get('rate_type', 'neg_binom')
    
    # ----------------------------------------------------------------------
    # 3) 입력 데이터에서 현재 data_type만 필터링
    #    - 이후 우도/공변량 계산은 이 슬라이스를 기준으로 수행
    # ----------------------------------------------------------------------
    input_data = sd['input_data']
    input_data_dt = input_data[input_data['data_type'] == data_type].copy()

    # ----------------------------------------------------------------------
    # 4) 표준오차(SE)와 유효표본크기(ESS) 보정
    #    - SE ≤ 0 또는 결측: (UCI-LCI)/(2*1.96)로 대체
    #    - ESS 결측/음수:   이항근사 p(1-p)/SE^2 로 대체
    #    - 보정치(카운트)를 로그로 알려줌
    # ----------------------------------------------------------------------
    invalid_se_mask   = (input_data_dt['standard_error'] < 0) | (input_data_dt['standard_error'].isna())
    se_replacement    = (input_data_dt['upper_ci'] - input_data_dt['lower_ci']) / (2 * 1.96)
    se                = input_data_dt['standard_error'].mask(invalid_se_mask, se_replacement)
    num_se_augmented  = int(invalid_se_mask.sum())

    invalid_ess_mask  = (input_data_dt['effective_sample_size'] < 0) | (input_data_dt['effective_sample_size'].isna())
    ess_replacement   = input_data_dt['value'] * (1 - input_data_dt['value']) / se**2
    ess               = input_data_dt['effective_sample_size'].mask(invalid_ess_mask, ess_replacement)
    num_ess_augmented = int(invalid_ess_mask.sum())

    input_data_dt['standard_error']        = se
    input_data_dt['effective_sample_size'] = ess
    print(f"Standard errors replaced: {num_se_augmented}")
    print(f"Effective sample sizes filled: {num_ess_augmented}")

    # ----------------------------------------------------------------------
    # 5) 공유데이터에 슬라이스 저장 및 데이터 존재 여부 플래그
    #    - 이후 단계(스플라인/공변량/우도)에서 사용
    # ----------------------------------------------------------------------
    sd[f'input_data_{data_type}'] = input_data_dt
    has_data = len(input_data_dt) > 0

    ############# I. Generate PYMC objects #########################################################
    with pm_model:
        # --- ages ---
        ages = np.asarray(sd.get('ages', sd['parameters']['ages']), dtype=np.int32)
        pm_model.add_coord("age", ages, mutable=False)

        ############ Calculating constrained_mu_age #########################################################
        if mu_age is not None:
            unconstrained_mu_age = mu_age
        else:
            unconstrained_mu_age = spline.spline(data_type)

        constrained_mu_age = priors.level_constraints(data_type, unconstrained_mu_age=unconstrained_mu_age)
        priors.derivative_constraints(data_type, mu_age=constrained_mu_age)            

        if mu_age_parent is not None: # penalize based on similarity to parent
            priors.similar(
                child_curve         = constrained_mu_age,
                parent_curve        = mu_age_parent,
                sigma_parent     = sigma_age_parent,
                sigma_diff_log = 0.0,
                eps              = 1e-9,
                penalty_name     = "_mu_age_parent_not_none"
            )

        ############ Calculating Pi #########################################################
        if has_data:
            mu_interval = age_groups.age_standardize_approx(data_type, mu_age=constrained_mu_age)
            # mu_interval = age_groups.age_standardize_approx(data_type, mu_age=unconstrained_mu_age)
            if include_covariates:
                pi = covariates.mean_covariate_model(data_type, mu_interval)
            else:
                pi = mu_interval
            # pm.Deterministic(f'constrained_mu_age_{data_type}', unconstrained_mu_age)
            # print('hello')

            ############ Likelihood based on rate_type #########################################################
            if rate_type == 'poisson':
                likelihood.poisson(data_type, input_data_dt, pi)

            elif rate_type == 'normal':
                sigma = pm.Uniform(
                    name=f'sigma_{data_type}',
                    lower=1e-4,
                    upper=1e-1,
                    initval=1e-2
                )
                likelihood.normal(data_type, input_data_dt, pi, sigma)

            elif rate_type == 'log_normal':
                sigma = pm.Uniform(
                    name=f'sigma_{data_type}',
                    lower=1e-4,
                    upper=1.0,
                    initval=1e-2
                )
                likelihood.log_normal(data_type, input_data_dt, pi, sigma)

            elif rate_type == 'offset_log_normal':
                sigma= pm.Uniform(
                    name=f'sigma_{data_type}',
                    lower=1e-4,
                    upper=10.0,
                    initval=1e-2
                )
                likelihood.offset_log_normal(data_type, input_data_dt, pi, sigma)

            elif rate_type == 'binom':
                likelihood.binom(data_type, input_data_dt, pi)

            elif rate_type == 'neg_binom':
                hetero = parameters.get('heterogeneity', None)
                lower = {'Slightly': 9.0, 'Moderately': 3.0, 'Very': 1.0}.get(hetero, 1.0)
                if data_type == 'pf':
                    lower = 1e12
                delta = covariates.dispersion_covariate_model(data_type, delta_lb=lower, delta_ub=lower * 9.0)
                likelihood.neg_binom(data_type, input_data_dt, pi, delta)     

            elif rate_type == 'beta_binom':
                hetero = parameters.get('heterogeneity', None)
                lower = {'Slightly': 9.0, 'Moderately': 3.0, 'Very': 1.0}.get(hetero, 1.0)
                if data_type == 'pf':
                    lower = 1e12
                delta = covariates.dispersion_covariate_model(data_type, delta_lb=lower, delta_ub=lower * 9.0)
                likelihood.beta_binom(data_type, input_data_dt, pi, delta)

            else:
                raise ValueError(f'Unsupported rate_type "{rate_type}"')

        else:
            if include_covariates:
                pi = covariates.mean_covariate_model(data_type, mu=None)
            else:
                assert False, "shouldn't be here"

        ############ Covariate Level Constraints #########################################################
        if include_covariates:
            priors.covariate_level_constraints(data_type, constrained_mu_age)

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
    draws=2000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.9,
    use_advi=False,
    use_metropolis=True,        # ← 추가
    vi_iters=20000,
    vi_lr=1e-3,
    verbose=False,
    nuts_max_treedepth=10,      # ← NUTS 전용
):
    t_start = time.time()
    logger = logging.getLogger(__name__)
    if verbose:
        logging.basicConfig(level=logging.INFO)

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
            return idata  # ADVI면 여기서 종료

        if use_metropolis:
            if verbose:
                logger.info("  ▶ Metropolis 샘플링 수행 중...")
            step = pm.Metropolis()
            # ⚠ NUTS 전용 인자는 절대 넣지 말 것 (nuts/nuts_sampler_kwargs 등)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                step=step,
                initvals=map_estimate,
                return_inferencedata=True,
                progressbar=verbose,
                # target_accept는 무시되지만 에러는 안 남 — 원한다면 빼도 무방
            )
        else:
            if verbose:
                logger.info("  ▶ NUTS 샘플링 수행 중...")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                initvals=map_estimate,
                target_accept=target_accept,
                nuts_sampler_kwargs={"max_treedepth": nuts_max_treedepth},
                return_inferencedata=True,
                progressbar=verbose,
            )

    if verbose:
        logger.info(f"[asr] 전체 소요 시간: {time.time()-t_start:.1f}초")
    return idata


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
    sd = pm_model.shared_data
    G  = sd['region_id_graph']
    global_id = sd['global_id']  # 경로 계산은 id 기준으로

    # -------------------- 0) baseline mu_age (draw x age) --------------------
    arr = idata.posterior['constrained_mu_age_p'].values  # (chain, draw, age)
    n_chain, n_draw, n_ages = arr.shape
    mu_trace = arr.reshape((n_chain * n_draw, n_ages))     # (n_samples, n_ages)
    n_samples = mu_trace.shape[0]

    ages = np.asarray(pm_model.coords["age"], dtype=int)
    assert len(ages) == n_ages, f"Age axis mismatch: len(ages)={len(ages)} vs n_ages={n_ages}"

    # 공용 데이터
    detailed_pop = sd['detailed_pop']

    # -------------------- 1) 공변량/RE 미포함 모드 --------------------
    if not include_covariates:
        # leaf(국가) 수집
        loc = int(location_id)
        if loc in G:
            leaf_ids = [n for n in nx.bfs_tree(G, loc) if G.out_degree(n) == 0] or [loc]
        else:
            leaf_ids = [loc]

        if return_scalar:
            num_prev  = np.zeros(n_samples)
            num_cases = np.zeros(n_samples)
            den = 0.0

            for leaf in leaf_ids:
                w = _pop_weights_for_leaf(detailed_pop, leaf, year_id, ages, sex_name)
                ws = w.sum()
                if ws <= 0:
                    continue
                num_prev  += (mu_trace * w[None, :]).sum(axis=1)
                num_cases += (mu_trace * w[None, :]).sum(axis=1)
                den += ws

            if not np.isfinite(den) or den <= 0:
                raise ValueError(f"[predict_for] detailed_pop empty: loc={location_id}, year={year_id}, sex={sex_name}")

            prevalence = np.clip(num_prev / den, lower, upper)
            cases      = num_cases
            return {"prevalence": prevalence, "cases": cases}
        else:
            return np.clip(mu_trace, lower, upper)

    # -------------------- 2) 공변량/RE 포함 모드 --------------------
    # data_type 추정(공유된 값 사용)
    dt = sd.get('data_type', 'p')

    # U / U_ref (RE 매트릭스)
    U     = sd[f'U_{dt}']           # (n_obs, n_re)
    U_ref = sd[f'U_ref_{dt}']       # shift vector (index=U.columns)

    # X (표준화된 디자인) + 센터링/스케일링
    X           = sd[f'X_{dt}']                 # (n_obs, n_cov)
    X_centering = sd[f'X_centering_{dt}']
    X_scaling   = sd[f'X_scaling_{dt}']

    # alpha/beta posterior trace (벡터)
    def _trace_vec(varname):
        if varname in idata.posterior:
            v = idata.posterior[varname].values  # (chain, draw, dim)
            return v.reshape(n_chain * n_draw, v.shape[-1])
        return None

    alpha_name = f'alpha_{dt}'
    beta_name  = f'beta_{dt}'

    alpha_trace = _trace_vec(alpha_name)
    beta_trace  = _trace_vec(beta_name)

    # 존재 가드(없으면 0벡터로)
    if alpha_trace is None and U.shape[1] > 0:
        alpha_trace = np.zeros((n_samples, U.shape[1]), dtype=float)
    if beta_trace is None and X.shape[1] > 0:
        beta_trace = np.zeros((n_samples, X.shape[1]), dtype=float)

    # leaf nodes
    loc = int(location_id)
    if loc in G:
        leaf_ids = [n for n in nx.bfs_tree(G, loc) if G.out_degree(n) == 0] or [loc]
    else:
        leaf_ids = [loc]

    # output 템플릿에서 (leaf, sex, year) covariate row 가져오기
    output_tpl = sd['output_template'].copy()
    output_tpl["location_id"] = output_tpl["location_id"].astype(int)
    output_tpl["sex_name"]    = output_tpl["sex_name"].astype(str)
    output_tpl["year_id"]     = output_tpl["year_id"].astype(int)
    grp = output_tpl.set_index(["location_id","sex_name","year_id"]).sort_index()

    # X_df: 훈련과 동일한 열/순서(좌표)로 필터링 + 성별 적용 + 표준화
    cov_dim = f"fe_eff_name_{dt}"
    cov_cols = list(pm_model.coords.get(cov_dim, X.columns.to_list()))
    if isinstance(X, pd.DataFrame) and not X.empty:
        X_df = grp.filter(cov_cols, axis=1).copy()
        if "x_sex" in cov_cols:
            sex_map = {'Male': .5, 'Both': 0., 'Female': -.5}
            X_df["x_sex"] = sex_map.get(sex_name, 0.0)
        # 표준화: (x - mean) / std  (브로드캐스트는 인덱스 정렬로 자동 정렬)
        X_df = (X_df - X_centering)[cov_cols] / X_scaling[cov_cols]
    else:
        X_df = pd.DataFrame(index=grp.index)

    # U_row 템플릿
    if isinstance(U, pd.DataFrame) and not U.empty:
        re_cols = list(U.columns)
        U_row_template = pd.Series(0.0, index=re_cols)
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
        # (a) U_row (경로 → 중심화 적용)
        if not U_row_template.empty and (leaf in G):
            U_row = U_row_template.copy()
            path = nx.shortest_path(G, global_id, leaf)
            for node in path[1:]:
                if node in U_row.index:
                    U_row[node] = 1.0 - U_ref.get(node, 0.0)
        else:
            U_row = pd.Series(dtype=float)

        # (b) log_shift = alpha·U + beta·x
        if alpha_trace is not None and not U_row.empty:
            # (n_samples, n_re) · (n_re,) → (n_samples,)
            log_shift = alpha_trace.dot(U_row.values)
        else:
            log_shift = np.zeros(n_samples)

        if (beta_trace is not None) and ((leaf, sex_name, year_id) in X_df.index):
            x_vals = X_df.loc[(leaf, sex_name, year_id), cov_cols].values
            # (n_samples, n_cov) · (n_cov,) → (n_samples,)
            log_shift = log_shift + beta_trace.dot(x_vals)

        # (c) 예측 곡선
        preds_leaf = np.clip(mu_trace * np.exp(log_shift)[:, None], lower, upper)

        # (d) 성별 포함 strict 가중치
        w = _pop_weights_for_leaf(detailed_pop, leaf, year_id, ages, sex_name)
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
        if not np.isfinite(den) or den <= 0:
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