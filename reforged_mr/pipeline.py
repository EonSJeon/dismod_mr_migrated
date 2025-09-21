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
        verbose              = False
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
    if verbose:
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
            if include_covariates:
                pi = covariates.mean_covariate_model(data_type, mu_interval)
            else:
                pi = mu_interval

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

# def return_map_estimate(pm_model, verbose=False):
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     verbose = True

#     with pm_model:
#         if verbose:
#             logger.info("  ▶ pm.find_MAP() 수행 중...")
        
#         map_estimate = pm.find_MAP()
#     return map_estimate

def return_map_estimate(
    pm_model,
    *,
    outer_reps: int = 3,
    verbose: bool = True,
    methods = None,
):
    """
    Robust stage-wise MAP:
      [Spline -> RE -> Spline -> FE -> Spline -> Dispersion] × outer_reps
    - Early stages: Powell (derivative-free; SciPy fmin_powell 계열)
    - Final full MAP: L-BFGS-B
    - RE 단계: 계층(BFS)로 묶어 최적화, 비어 있으면 전체 alpha 일괄 최적화로 폴백
    - FE 단계: 1개씩 누적 순차 적합(안정화)
    - NaN/실패 시 자동 대체 방법으로 재시도
    """
    import logging, re, numpy as np
    import pymc as pm
    import networkx as nx

    # ---------------- logger ----------------
    logger = logging.getLogger("pipeline")
    if verbose:
        logger.setLevel(logging.INFO)

    # ------------- method policy ------------
    default_methods = {
        "spline": "Powell",
        "re": "Powell",
        "re_hyper": "L-BFGS-B",
        "fe": "Powell",
        "disp": "L-BFGS-B",
        "final": "L-BFGS-B",
    }
    if methods:
        default_methods.update(methods)

    fallback_chain = ["Powell", "Nelder-Mead", "L-BFGS-B"]

    def _fit_with_fallback(stage_vars, desc, prefer):
        """pm.find_MAP(vars=stage_vars) with method fallback & start accumulation"""
        nonlocal start_point, last_good_point, last_good_logp

        if not stage_vars:
            if verbose:
                logger.info(f"[SKIP] {desc} (no matching free RVs)")
            return

        tried = []
        for mth in ([prefer] if prefer else []) + [x for x in fallback_chain if x != prefer]:
            if mth in tried:
                continue
            tried.append(mth)
            if verbose:
                logger.info(f"[MAP] {desc} ({mth}): {', '.join(v.name for v in stage_vars)}")
            try:
                point = pm.find_MAP(
                    vars=stage_vars,
                    start=start_point,
                    method=mth,
                    progressbar=verbose   # <-- 여기만 추가
                )
                # merge into rolling start
                start_point = point if start_point is None else {**start_point, **point}
                # optional: score logp
                try:
                    logp_fn = pm_model.compile_logp()
                    lp = float(logp_fn(start_point))
                    last_good_point, last_good_logp = start_point.copy(), lp
                except Exception:
                    pass
                return
            except Exception as e:
                if verbose:
                    logger.info(f"[WARN] {desc} with {mth} failed: {e}")
                continue

        # if all fail, keep going with current start_point
        if verbose:
            logger.info(f"[WARN] {desc} failed for all methods. Continuing with previous start.")


    with pm_model:
        sd = getattr(pm_model, "shared_data", {})
        dt = sd.get("data_type", "")
        g  = sd.get("region_id_graph", None)
        root = sd.get("reference_area_id", sd.get("global_id", None))

        free = list(pm_model.free_RVs)
        names = {v.name: v for v in free}

        def _pick_by_prefix(prefixes):
            return [v for v in free if any(v.name.startswith(p) for p in prefixes)]

        def _pick_by_regex(patterns):
            regs = [re.compile(p) for p in patterns]
            out = []
            for v in free:
                if any(r.search(v.name) for r in regs):
                    out.append(v)
            return out

        # ---------- group discovery ----------
        gamma_vars = _pick_by_regex([
            rf"^gamma(_{dt}(_\d+)?|_{dt}$|\b)",
            rf"^mu_age(_{dt}\b|_{dt}_\d+|\b)",
            rf"spline.*{dt}",
        ])

        # FE
        beta_vars = _pick_by_prefix([f"beta_{dt}_"]) if dt else _pick_by_regex([r"^beta_"])

        # RE
        alpha_vars = _pick_by_prefix([f"alpha_{dt}_"]) if dt else _pick_by_regex([r"^alpha_"])
        sigma_alpha_vars = _pick_by_prefix([f"sigma_alpha_{dt}_"]) if dt else _pick_by_regex([r"^sigma_alpha_"])

        # Dispersion
        disp_vars = []
        if dt:
            if f"eta_{dt}" in names:  disp_vars.append(names[f"eta_{dt}"])
            if f"zeta_{dt}" in names: disp_vars.append(names[f"zeta_{dt}"])
        disp_vars += [v for v in free if v.name.startswith("eta_") or v.name.startswith("zeta_")]
        # de-dup
        seen = set(); disp_vars = [v for v in disp_vars if (v.name not in seen and not seen.add(v.name))]

        # anchors (optional)
        anchors = [v for v in beta_vars if v.name.endswith(("x_intercept", "_intercept", "intercept"))]

        # --------- diagnostics print ---------
        if verbose:
            def _nm(vs): return [v.name for v in vs]
            logger.info(f"[VAR] gammas={_nm(gamma_vars)}")
            logger.info(f"[VAR] betas ={_nm(beta_vars)}")
            logger.info(f"[VAR] alphas={_nm(alpha_vars)}")
            logger.info(f"[VAR] s.alp ={_nm(sigma_alpha_vars)}")
            logger.info(f"[VAR] disp  ={_nm(disp_vars)}")

        # rolling start + best point
        start_point = None
        last_good_point, last_good_logp = None, -np.inf

        # ---------- stages ----------
        def stage_spline():
            _fit_with_fallback(gamma_vars + anchors, "spline (monolithic)", default_methods["spline"])

        def stage_fe():
            # sequential cumulative FE (stable)
            if not beta_vars:
                return
            # intercept 먼저, 나머지는 이름순
            intercepts = [v for v in beta_vars if v in anchors]
            others     = sorted([v for v in beta_vars if v not in anchors], key=lambda x: x.name)
            ordered = intercepts + others
            for k in range(1, len(ordered) + 1):
                _fit_with_fallback(ordered[:k], f"fixed effects (first {k}/{len(ordered)})", default_methods["fe"])

        def stage_re_bfs_or_bulk():
            if not alpha_vars:
                if verbose:
                    logger.info("[RE] No alpha_* free RVs found; skip BFS and hyper will still be tuned.")
            else:
                if g is not None and root is not None:
                    # map node-id string -> alpha RV
                    name2alpha = {}
                    prefix = f"alpha_{dt}_" if dt else "alpha_"
                    for v in alpha_vars:
                        suf = v.name.split(prefix, 1)[-1]
                        name2alpha[suf] = v
                    # BFS parent + children grouping
                    try:
                        order = list(nx.bfs_tree(g, root))
                    except Exception:
                        order = list(getattr(g, "nodes", lambda: [])())
                    hit = 0
                    for p in order:
                        succ = list(g.successors(p)) if hasattr(g, "successors") else []
                        group = []
                        for nd in list(succ) + [p]:
                            key = str(nd)
                            if key in name2alpha:
                                group.append(name2alpha[key])
                        if group:
                            hit += 1
                            _fit_with_fallback(group + anchors, f"random effects (parent {p} + children)", default_methods["re"])
                    # 폴백: BFS에서 한 번도 hit 못 하면 전체 alpha를 일괄 최적화
                    if hit == 0:
                        _fit_with_fallback(alpha_vars + anchors, "random effects (bulk)", default_methods["re"])
                else:
                    # 그래프가 없으면 일괄
                    _fit_with_fallback(alpha_vars + anchors, "random effects (bulk)", default_methods["re"])

            # hyper (항상 시도)
            if sigma_alpha_vars:
                _fit_with_fallback(sigma_alpha_vars + anchors, "RE hyper (sigma_alpha_*)", default_methods["re_hyper"])

        def stage_disp():
            if disp_vars:
                _fit_with_fallback(disp_vars, "dispersion (eta_/zeta_*)", default_methods["disp"])

        # --------- orchestrate ----------
        if verbose:
            logger.info("▶ Stage-wise MAP initialization started")

        for rep in range(outer_reps):
            if verbose:
                logger.info(f"— Outer loop {rep+1}/{outer_reps}")
            stage_spline()
            stage_re_bfs_or_bulk()
            stage_spline()
            stage_fe()
            stage_spline()
            stage_disp()

        # --------- final full MAP ----------
        if verbose:
            logger.info("▶ Final full MAP")
        try:
            final_map = pm.find_MAP(
                start=start_point, 
                method=default_methods["final"],
                progressbar=verbose,
            )
        except Exception as e:
            if verbose:
                logger.info(f"[WARN] Final MAP failed: {e}. Returning best start so far.")
            final_map = start_point if start_point is not None else {}
        if verbose:
            logger.info("✓ MAP initialization finished.")
        return final_map
#

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
    dt,
    pm_model,
    idata, 
    root_area           = 'Global',
    root_sex            = 'Both',
    root_year           = 'all',
    location_id         = None,
    sex_name            = 'Both',
    year_id             = 2005,
    population_weighted = True,
    lower               = 0.0,
    upper               = 1.0,
):
    sd = pm_model.shared_data
    params = sd['parameters']
    params_dt = params[dt]
    include_covariates = params_dt['include_covariates']
    G  = sd['region_id_graph']
    global_id = sd['global_id']  # 경로 계산은 id 기준으로
    if location_id is None:
        location_id = global_id

    # -------------------- 0) baseline mu_age (draw x age) --------------------
    arr = idata.posterior[f'constrained_mu_age_{dt}'].values  # (chain, draw, age)
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

        return np.clip(mu_trace, lower, upper)
            

    # -------------------- 2) 공변량/RE 포함 모드 --------------------

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

        if population_weighted:
                num += preds_leaf * w[None, :]
                den += w
        else:
            num += preds_leaf
            leaf_count += 1
            

    # finalize
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
    ages = np.asarray(ages, dtype=int)

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
#
def _ci(samples, coverage):
    alpha = 100.0 - coverage
    lo = float(np.percentile(samples, alpha/2.0))
    hi = float(np.percentile(samples, 100.0 - alpha/2.0))
    return lo, hi

def world_predict(
    dt,
    pm_model,
    idata,
    years,
    target_levels={0, 2, 3},
    sexes=('Both', 'Male', 'Female'),
    lower=0.0,
    upper=1.0,
    output_csv_path=None,
    verbose=False,
):
    sd = pm_model.shared_data
    params = sd['parameters']
    params_dt = params[dt]
    include_covariates = params_dt['include_covariates']
    G  = sd['region_id_graph']

    id_to_name     = sd['id_to_name']
    ages = np.asarray(pm_model.coords["age"], dtype=int)
    age_weights = params['age_weights']
    age_w_std          = _as_age_weight_vector(age_weights, ages)
    dtype          = np.float64  # 필요 시 mu_trace.dtype로 바꿔도 됨

    # ---- posterior shapes ----
    arr = idata.posterior[f'constrained_mu_age_{dt}'].values  # (chain, draw, age)
    n_chain, n_draw, n_ages = arr.shape
    mu_trace = arr.reshape((n_chain * n_draw, n_ages)).astype(dtype, copy=False)
    n_samples = mu_trace.shape[0]
    assert n_ages == len(ages), f"Age axis mismatch: {n_ages} vs {len(ages)}"

    def _trace_vec(varname):
        if varname in idata.posterior:
            v = idata.posterior[varname].values  # (chain, draw, dim)
            return v.reshape(n_chain * n_draw, v.shape[-1]).astype(dtype, copy=False)
        return None

    U      = sd.get(f'U_{dt}',      None)  # DataFrame or None
    U_ref  = sd.get(f'U_ref_{dt}',  None)  # dict/Series or None
    X      = sd.get(f'X_{dt}',      None)  # DataFrame or None
    X_ctr  = sd.get(f'X_centering_{dt}', pd.Series(dtype=float))
    X_scl  = sd.get(f'X_scaling_{dt}',   pd.Series(dtype=float))
    outtpl = sd['output_template'].copy()
    outtpl["location_id"] = outtpl["location_id"].astype(int)
    outtpl["sex_name"]    = outtpl["sex_name"].astype(str)
    outtpl["year_id"]     = outtpl["year_id"].astype(int)
    grp = outtpl.set_index(["location_id", "sex_name", "year_id"]).sort_index()

    cov_dim  = f"fe_eff_name_{dt}"
    cov_cols = list(pm_model.coords.get(cov_dim, X.columns.to_list())) if isinstance(X, pd.DataFrame) else []

    alpha = _trace_vec(f'alpha_{dt}')
    beta  = _trace_vec(f'beta_{dt}')
    if include_covariates:
        if (alpha is None) and isinstance(U, pd.DataFrame) and U.shape[1] > 0:
            alpha = np.zeros((n_samples, U.shape[1]), dtype=dtype)
        if (beta  is None) and isinstance(X, pd.DataFrame) and X.shape[1] > 0:
            beta  = np.zeros((n_samples, X.shape[1]), dtype=dtype)
    else:
        alpha = None
        beta  = None

    # ---- 타깃 노드 목록(level 0,2,3) ----
    nodes = []
    target_nodes_g = set()
    for nid_g, data in G.nodes(data=True):
        lvl = data.get('level', None)
        if lvl in target_levels:
            nodes.append((nid_g, lvl, id_to_name.get(int(nid_g) if str(nid_g).isdigit() else nid_g, str(nid_g))))
            target_nodes_g.add(nid_g)
    # 정렬은 보기 편하게
    nodes.sort(key=lambda x: (x[1], str(x[2])))

    # ---- 리프 & 조상(타깃) 매핑 ----
    global_id = sd['global_id']  # 그래프 노드 타입 유지
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]

    ancestors_targets = {}  # key: leaf_int, val: [ancestor nodes in graph id space]
    for leaf_g in leaves:
        try:
            path = nx.shortest_path(G, global_id, leaf_g)
        except nx.NetworkXNoPath:
            path = [leaf_g]
        anc = [n for n in path if n in target_nodes_g]
        # leaf 자신이 타깃이면 포함
        if (leaf_g in target_nodes_g) and (leaf_g not in anc):
            anc.append(leaf_g)
        leaf_int = int(leaf_g) if str(leaf_g).isdigit() else leaf_g
        ancestors_targets[leaf_int] = anc

    # ---- U_row (리프마다 RE 경로 벡터) 사전 계산 ----
    if isinstance(U, pd.DataFrame) and not U.empty:
        re_cols = list(U.columns)
        U_ref_s = pd.Series(U_ref).reindex(re_cols).fillna(0.0)
        U_row_map = {}
        for leaf_g in leaves:
            u = pd.Series(0.0, index=re_cols)
            try:
                path = nx.shortest_path(G, global_id, leaf_g)
            except nx.NetworkXNoPath:
                path = [leaf_g]
            for node in path[1:]:
                if node in u.index:
                    u.loc[node] = 1.0 - float(U_ref_s.get(node, 0.0))
            leaf_int = int(leaf_g) if str(leaf_g).isdigit() else leaf_g
            U_row_map[leaf_int] = u.to_numpy(dtype=dtype, copy=False)
    else:
        U_row_map = {int(l) if str(l).isdigit() else l: None for l in leaves}

    # ---- 캐시 ----
    X_ctr = pd.Series(X_ctr, copy=False)
    X_scl = pd.Series(X_scl, copy=False)
    x_cache = {}  # (leaf_int, sex, year) -> np.ndarray or None
    w_cache = {}  # (leaf_int, sex, year) -> np.ndarray or None

    # sex_id_map = {'Male': 1, 'Female': 2, 'Both': 3}
    rows = []

    # ---- 진행률 카운터 준비 ----
    total_steps = len(years) * len(sexes) * len(leaves)
    step = 0

    for year in years:
        year = int(year)
        for sex in sexes:
            if verbose:
                print(f"[world_predict] Processing year={year}, sex={sex}")

            cases_by_age_accum = {}
            den_by_age         = {}

            for leaf_g in leaves:
                step += 1
                progress = 100.0 * step / total_steps

                leaf_int = int(leaf_g) if str(leaf_g).isdigit() else leaf_g
                target_ancs = ancestors_targets.get(leaf_int, [])
                if not target_ancs:
                    continue

                # ---- 인구 가중치 ----
                w_key = (leaf_int, sex, year)
                if w_key in w_cache:
                    w = w_cache[w_key]

                else:
                    try:
                        w = _pop_weights_for_leaf(sd['detailed_pop'], int(leaf_int), year, ages, sex)

                    except Exception as e:
                        if verbose:
                            print(f"  skip leaf={leaf_int}, sex={sex}, year={year} :: {e} "
                                  f"({progress:.1f}% done)")
                        w_cache[w_key] = None
                        continue
                    w_cache[w_key] = w
                if w is None or (not np.isfinite(w).all()) or (w.sum() <= 0):
                    continue

                # ---- log_shift ----
                if include_covariates:
                    log_shift = np.zeros(n_samples, dtype=dtype)
                    u_vals = U_row_map.get(leaf_int, None)
                    if (alpha is not None) and (u_vals is not None):
                        log_shift += alpha.dot(u_vals)

                    x_vals = x_cache.get(w_key, None)
                    if (x_vals is None) and isinstance(X, pd.DataFrame) and (len(cov_cols) > 0):
                        if (int(leaf_int), sex, year) in grp.index:
                            row = grp.loc[(int(leaf_int), sex, year)].reindex(cov_cols)
                            if "x_sex" in cov_cols:
                                sex_map = {'Male': .5, 'Both': 0., 'Female': -.5}
                                row.loc["x_sex"] = sex_map.get(sex, 0.0)
                            xc = X_ctr.reindex(cov_cols).fillna(0.0)
                            xs = X_scl.reindex(cov_cols).replace(0, 1.0).fillna(1.0)
                            x_vals = ((row - xc) / xs).to_numpy(dtype=dtype)
                        x_cache[w_key] = x_vals
                    if (beta is not None) and (x_vals is not None):
                        log_shift += beta.dot(x_vals)
                else:
                    log_shift = np.zeros(n_samples, dtype=dtype)

                preds_leaf = mu_trace * np.exp(log_shift)[:, None]
                if (lower != 0.0) or (upper != 1.0):
                    np.clip(preds_leaf, lower, upper, out=preds_leaf)
                cases_by_age = preds_leaf * w[None, :]

                # ---- 조상 타깃 노드에 누적 ----
                for node_g in target_ancs:
                    if node_g not in cases_by_age_accum:
                        cases_by_age_accum[node_g] = np.zeros((n_samples, n_ages), dtype=dtype)
                        den_by_age[node_g] = np.zeros(n_ages, dtype=dtype)
                    cases_by_age_accum[node_g] += cases_by_age
                    den_by_age[node_g]         += w
                    if verbose:
                        loc_name = id_to_name.get(
                            int(node_g) if str(node_g).isdigit() else node_g,
                            str(node_g)
                        )
                        print(f"    leaf={leaf_int} → node={node_g} ({loc_name}), "
                              f"year={year}, sex={sex} :: {progress:.1f}% done")

            # summarize per node
            for node_g, level, loc_name in nodes:
                if node_g not in cases_by_age_accum:
                    if verbose:
                        print(f"[world_predict] No data for node={node_g} ({loc_name}), sex={sex}, year={year}")
                    continue
                cases_mat = cases_by_age_accum[node_g]
                den_vec   = den_by_age[node_g]
                den_total = float(den_vec.sum())
                if (den_total <= 0) or (not np.isfinite(den_total)):
                    if verbose:
                        print(f"[world_predict] Zero denom for node={node_g} ({loc_name}), sex={sex}, year={year}")
                    continue

                total_cases  = cases_mat.sum(axis=1)
                prev_samples = total_cases / den_total
                if (lower != 0.0) or (upper != 1.0):
                    prev_samples = np.clip(prev_samples, lower, upper)

                den_safe   = np.where(den_vec > 0, den_vec, 1e-12)
                prev_curve = cases_mat / den_safe[None, :]
                if (lower != 0.0) or (upper != 1.0):
                    np.clip(prev_curve, lower, upper, out=prev_curve)
                prev_std_samples = prev_curve @ age_w_std

                # === 추가: 표준편차 & 여러 신뢰구간 ===
                
                coverages = [
                    (100.0-5/1.0,     "p95"),
                    (100.0-5/2.0,     "p97_5"),          # = 100 - 5/2
                    (100.0-5/3.0, "p98_333"),     # ≈ 98.333…
                    (100.0-5/4.0,    "p98_75"),         # = 100 - 5/4
                ]

                # 표준편차
                sd_prev      = float(np.std(prev_samples, ddof=1))
                sd_cases     = float(np.std(total_cases, ddof=1))
                sd_prev_std  = float(np.std(prev_std_samples, ddof=1))

                # 기본 필드
                row = {
                    "location_id":   int(node_g) if str(node_g).isdigit() else node_g,
                    "location_name": loc_name,
                    "sex_name":      sex,
                    "sex_id":        {'Male':1,'Female':2,'Both':3}[sex],
                    "level":         int(level) if level is not None else None,
                    "year":          year,

                    # mean
                    "mean_prev":     float(np.mean(prev_samples)),
                    "mean_cases":    float(np.mean(total_cases)),
                    "mean_prev_std": float(np.mean(prev_std_samples)),

                    # sd
                    "sd_prev":       sd_prev,
                    "sd_cases":      sd_cases,
                    "sd_prev_std":   sd_prev_std,
                }

                # 각 coverage별 CI 추가
                for cov, tag in coverages:
                    lo_p, hi_p = _ci(prev_samples, cov)
                    lo_c, hi_c = _ci(total_cases,  cov)
                    lo_s, hi_s = _ci(prev_std_samples, cov)
                    row[f"lower_prev_{tag}"]     = lo_p
                    row[f"upper_prev_{tag}"]     = hi_p
                    row[f"lower_cases_{tag}"]    = lo_c
                    row[f"upper_cases_{tag}"]    = hi_c
                    row[f"lower_prev_std_{tag}"] = lo_s
                    row[f"upper_prev_std_{tag}"] = hi_s

                rows.append(row)

    if len(rows) == 0:
        print("[world_predict] Warning: no rows computed; CSV not written.")
        return pd.DataFrame(columns=[
            "location_id","location_name","sex_name","sex_id","level","year",
            "mean_prev","lower_prev_p95","upper_prev_p95",  # 헤더 예시 (비어있을 수도)
        ])
    
    df = (
        pd.DataFrame(rows)
        .sort_values(["year","level","location_name","sex_id"])
        .reset_index(drop=True)
    )
    df.to_csv(output_csv_path, index=False)
    print(f"[world_predict] Saved {len(df)} rows to '{output_csv_path}'")
    return df

#

# def world_predict(
#     pm_model,
#     idata,
#     years,              
#     output_csv_path
# ):
#     # 1) 그래프/이름 매핑
#     region_id_graph = pm_model.shared_data['region_id_graph']
#     id_to_name      = pm_model.shared_data['id_to_name']
#     ages            = pm_model.coords['age']
#     age_weights_in  = pm_model.shared_data['age_weights']

#     ages = np.asarray(ages, dtype=int)
#     age_w = _as_age_weight_vector(age_weights_in, ages)

#     # 2) level 0,2,3 노드만
#     target_levels = {0, 2, 3}
#     nodes = []
#     for nid_str, data in region_id_graph.nodes(data=True):
#         level = data.get('level', None)
#         if level in target_levels:
#             try:
#                 nid_int = int(nid_str)
#             except (TypeError, ValueError):
#                 continue
#             nodes.append((nid_int, level, id_to_name.get(nid_int, str(nid_int))))

#     # ✅ GBD 관행 sex_id 매핑
#     sex_id_map = {'Male': 1, 'Female': 2, 'Both': 3}

#     rows = []
#     for year in years:
#         for sex in ['Both', 'Male', 'Female']:
#             for loc_id, level, loc_name in nodes:
#                 try:
#                     # 1) 스칼라(유병률/환자수)
#                     res_scalar = predict_for(
#                         pm_model,
#                         idata,
#                         root_area='Global',
#                         root_sex='Both',
#                         root_year='all',
#                         location_id=loc_id,
#                         sex_name=sex,
#                         year_id=int(year),
#                         population_weighted=True,
#                         lower=0.0,
#                         upper=1.0,
#                         include_covariates=True,
#                         return_scalar=True,
#                     )
#                     prev_samples  = res_scalar["prevalence"]
#                     cases_samples = res_scalar["cases"]

#                     # 2) 연령표준화 유병률 (곡선 한 번 더)
#                     preds_curve = predict_for(
#                         pm_model,
#                         idata,
#                         root_area='Global',
#                         root_sex='Both',
#                         root_year='all',
#                         location_id=loc_id,
#                         sex_name=sex,
#                         year_id=int(year),
#                         population_weighted=True,
#                         lower=0.0,
#                         upper=1.0,
#                         include_covariates=True,
#                         return_scalar=False,
#                     )
#                     if preds_curve.ndim != 2 or preds_curve.shape[1] != len(age_w):
#                         raise ValueError(
#                             f"preds_curve shape {preds_curve.shape} != age_weights length {len(age_w)}"
#                         )
#                     prev_std_samples = preds_curve @ age_w  # (n_samples,)

#                     # 요약
#                     mean_prev     = float(np.mean(prev_samples))
#                     lower_prev    = float(np.percentile(prev_samples, 2.5))
#                     upper_prev    = float(np.percentile(prev_samples, 97.5))

#                     mean_cases    = float(np.mean(cases_samples))
#                     lower_cases   = float(np.percentile(cases_samples, 2.5))
#                     upper_cases   = float(np.percentile(cases_samples, 97.5))

#                     mean_prev_std   = float(np.mean(prev_std_samples))
#                     lower_prev_std  = float(np.percentile(prev_std_samples, 2.5))
#                     upper_prev_std  = float(np.percentile(prev_std_samples, 97.5))

#                     rows.append({
#                         "location_id":     loc_id,
#                         "location_name":   loc_name,
#                         "sex_name":        sex,
#                         "sex_id":          sex_id_map[sex],
#                         "level":           level,
#                         "year":            int(year),
#                         "mean_prev":       mean_prev,
#                         "lower_prev":      lower_prev,
#                         "upper_prev":      upper_prev,
#                         "mean_cases":      mean_cases,
#                         "lower_cases":     lower_cases,
#                         "upper_cases":     upper_cases,
#                         "mean_prev_std":   mean_prev_std,
#                         "lower_prev_std":  lower_prev_std,
#                         "upper_prev_std":  upper_prev_std,
#                     })

#                 except ValueError as e:
#                     print(f"[world_predict] Skip loc={loc_id} ({loc_name}), year={year}, sex={sex} :: {e}")
#                 except Exception as e:
#                     print(f"[world_predict] Error  loc={loc_id} ({loc_name}), year={year}, sex={sex} :: {e}")

#     # 4) 저장
#     if len(rows) == 0:
#         print("[world_predict] Warning: no rows computed; CSV not written.")
#         return pd.DataFrame(columns=[
#             "location_id","location_name","sex_name","sex_id","level","year",
#             "mean_prev","lower_prev","upper_prev",
#             "mean_cases","lower_cases","upper_cases",
#             "mean_prev_std","lower_prev_std","upper_prev_std",
#         ])

#     df = (
#         pd.DataFrame(rows)
#         .sort_values(["year","level","location_name","sex_id"])
#         .reset_index(drop=True)
#     )
#     df.to_csv(output_csv_path, index=False)
#     print(f"[world_predict] Saved {len(df)} rows to '{output_csv_path}'")

#     return df
# #

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