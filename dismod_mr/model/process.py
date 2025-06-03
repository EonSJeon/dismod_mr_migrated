import numpy as np
import pymc as pm
import dismod_mr
from dismod_mr.data import ModelVars
from dismod_mr.model import spline, priors, likelihood, covariates


def age_specific_rate(
    mr_model: dismod_mr.data.MRModel,
    data_type,
    reference_area='all',
    reference_sex='total',
    reference_year='all',
    mu_age=None,
    mu_age_parent=None,
    sigma_age_parent=None,
    rate_type='neg_binom',
    lower_bound=None,
    interpolation_method='linear',
    include_covariates=True,
    zero_re=False
):
    """
    Generate PyMC objects for epidemiological age-interval data model.
    """
    _data_type = data_type
    result = ModelVars()

    # 0) 부모 prior에 NaN 섞였으면 무시
    if isinstance(mu_age_parent, np.ndarray) and np.isnan(mu_age_parent).any() or \
       isinstance(sigma_age_parent, np.ndarray) and np.isnan(sigma_age_parent).any():
        mu_age_parent = None
        sigma_age_parent = None

    ages = np.array(mr_model.parameters['ages'])
    data = mr_model.get_data(data_type)
    lb_data = mr_model.get_data(lower_bound) if lower_bound else None
    parameters = mr_model.parameters.get(data_type, {})

    vars = ModelVars()
    vars['data'] = data

    # ── 1) Spline knots & smoothing ──────────────────────────────
    knots = np.array(
        parameters.get(
            'parameter_age_mesh',
            np.arange(ages[0], ages[-1] + 1, 5)
        )
    )
    smooth_map = {'No Prior': np.inf, 'Slightly': 0.5, 'Moderately': 0.05, 'Very': 0.005}
    smoothness_param = parameters.get('smoothness')
    if isinstance(smoothness_param, dict):
        amt = smoothness_param.get('amount')
        smoothing = float(amt) if isinstance(amt, (int, float)) else smooth_map.get(amt, 0.0)
    else:
        smoothing = smooth_map.get(smoothness_param, 0.0)

    pm_model = pm.Model()
    with pm_model:
        # 1-1) spline prior
        if mu_age is None:
            spline_vars = spline.spline(
                data_type=_data_type,
                ages=ages,
                knots=knots,
                smoothing=smoothing,
                interpolation_method=interpolation_method
            )
            vars.update(spline_vars)
        else:
            vars['mu_age'] = mu_age
            vars['ages'] = ages

        # 1-2) level & derivative constraints
        lc = priors.level_constraints(
            data_type=_data_type,
            parameters=parameters,
            unconstrained_mu_age=vars['mu_age'],
            ages=ages
        )
        vars.update(lc)

        dc = priors.derivative_constraints(
            data_type=_data_type,
            parameters=parameters,
            mu_age=vars['mu_age'],
            ages=ages
        )
        vars.update(dc)

        # 1-3) hierarchical similarity (부모 prior이 있을 때만)
        if mu_age_parent is not None:
            sim = priors.similar(
                data_type=_data_type,
                mu_child=vars['mu_age'],
                mu_parent=mu_age_parent,
                sigma_parent=sigma_age_parent,
                sigma_difference=0.0
            )
            vars.update(sim)

        # ── 2) age‐interval 평균 계산 ─────────────────────────────────
        if len(data) > 0:
            data = data.copy()
            # 2-1) standard_error, effective_sample_size 채우기
            se = data['standard_error'].mask(
                data['standard_error'] < 0,
                (data['upper_ci'] - data['lower_ci']) / (2 * 1.96)
            )
            ess = data['effective_sample_size'].fillna(
                data['value'] * (1 - data['value']) / se**2
            )
            data['standard_error'] = se
            data['effective_sample_size'] = ess

            age_int = dismod_mr.model.age_groups.age_standardize_approx(
                name=_data_type,
                age_weights=np.ones_like(vars['mu_age'].eval()),
                mu_age=vars['mu_age'],
                age_start=data['age_start'],
                age_end=data['age_end'],
                ages=ages
            )
            vars.update(age_int)

            # 2-2) covariate & pi
            if include_covariates:
                cov = covariates.mean_covariate_model(
                    data_type=_data_type,
                    mu=vars['mu_interval'],
                    input_data=data,
                    parameters=parameters,
                    model=mr_model,
                    root_area=reference_area,
                    root_sex=reference_sex,
                    root_year=reference_year,
                    zero_re=zero_re
                )
                vars.update(cov)
            else:
                vars['pi'] = vars['mu_interval']
        else:
            # 데이터가 없으면 covariate 모델만
            if include_covariates:
                cov = covariates.mean_covariate_model(
                    data_type=_data_type,
                    mu=None,
                    input_data=data,
                    parameters=parameters,
                    model=mr_model,
                    root_area=reference_area,
                    root_sex=reference_sex,
                    root_year=reference_year,
                    zero_re=zero_re
                )
                vars.update(cov)

        # ── 3) RATE_TYPE별 전처리 + likelihood 호출 ────────────────────
        if len(data) > 0:
            if rate_type == 'neg_binom':
                bad_ess = (data['effective_sample_size'] <= 0) | data['effective_sample_size'].isna()
                if bad_ess.any():
                    data.loc[bad_ess, 'effective_sample_size'] = 0.0

                big_ess = data['effective_sample_size'] >= 1e10
                if big_ess.any():
                    data.loc[big_ess, 'effective_sample_size'] = 1e10

                hetero = parameters.get('heterogeneity', None)
                lower = {'Slightly': 9.0, 'Moderately': 3.0, 'Very': 1.0}.get(hetero, 1.0)
                if data_type == 'pf':
                    lower = 1e12

                disp_cov = covariates.dispersion_covariate_model(
                    data_type=_data_type,
                    input_data=data,
                    delta_lb=lower,
                    delta_ub=lower * 9.0
                )
                vars.update(disp_cov)

                nb_dict = likelihood.neg_binom(
                    name=_data_type,
                    pi=vars['pi'],
                    delta=vars['delta'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                vars.update(nb_dict)

            elif rate_type == 'log_normal':
                missing = data['standard_error'] < 0
                if missing.any():
                    data.loc[missing, 'standard_error'] = 1e6

                vars['sigma'] = pm.Uniform(
                    name=f'sigma_{_data_type}',
                    lower=1e-4,
                    upper=1.0,
                    initval=1e-2
                )

                lg_dict = likelihood.log_normal(
                    name=_data_type,
                    pi=vars['pi'],
                    sigma=vars['sigma'],
                    p=data['value'].to_numpy(),
                    s=data['standard_error'].to_numpy()
                )
                vars.update(lg_dict)

            elif rate_type == 'normal':
                missing = data['standard_error'] < 0
                if missing.any():
                    data.loc[missing, 'standard_error'] = 1e6

                vars['sigma'] = pm.Uniform(
                    name=f'sigma_{_data_type}',
                    lower=1e-4,
                    upper=1e-1,
                    initval=1e-2
                )

                nm_dict = likelihood.normal(
                    name=_data_type,
                    pi=vars['pi'],
                    sigma=vars['sigma'],
                    p=data['value'].to_numpy(),
                    s=data['standard_error'].to_numpy()
                )
                vars.update(nm_dict)

            elif rate_type == 'binom':
                bad_ess = data['effective_sample_size'] < 0
                if bad_ess.any():
                    data.loc[bad_ess, 'effective_sample_size'] = 0.0

                bb_dict = likelihood.binom(
                    name=_data_type,
                    pi=vars['pi'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                vars.update(bb_dict)

            elif rate_type == 'beta_binom':
                bbb_dict = likelihood.beta_binom(
                    name=_data_type,
                    pi=vars['pi'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                vars.update(bbb_dict)

            elif rate_type == 'poisson':
                bad_ess = data['effective_sample_size'] < 0
                if bad_ess.any():
                    data.loc[bad_ess, 'effective_sample_size'] = 0.0

                pois_dict = likelihood.poisson(
                    name=_data_type,
                    pi=vars['pi'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                vars.update(pois_dict)

            elif rate_type == 'offset_log_normal':
                vars['sigma'] = pm.Uniform(
                    name=f'sigma_{_data_type}',
                    lower=1e-4,
                    upper=10.0,
                    initval=1e-2
                )

                oln_dict = likelihood.offset_log_normal(
                    name=_data_type,
                    pi=vars['pi'],
                    sigma=vars['sigma'],
                    p=data['value'].to_numpy(),
                    s=data['standard_error'].to_numpy()
                )
                vars.update(oln_dict)

            else:
                raise ValueError(f'Unsupported rate_type "{rate_type}"')

        # ── 4) 데이터가 전혀 없으면 likelihood 호출 없음 ─────────────────

        # ── 5) covariate‐level constraints ─────────────────────────────
        if include_covariates:
            clc = priors.covariate_level_constraints(
                data_type=_data_type,
                model=mr_model,
                vars=vars,
                ages=ages
            )
            vars.update(clc)

        # ── 6) lower‐bound 데이터 처리 (옵션) ──────────────────────────
        if lb_data is not None and len(lb_data) > 0:
            lb = {}
            lb_interval = dismod_mr.model.age_groups.age_standardize_approx(
                name=f'lb_{_data_type}',
                age_weights=np.ones_like(vars['mu_age'].eval()),
                mu_age=vars['mu_age'],
                age_start=lb_data['age_start'],
                age_end=lb_data['age_end'],
                ages=ages
            )
            lb.update(lb_interval)

            if include_covariates:
                lb_cov = covariates.mean_covariate_model(
                    data_type=f'lb_{_data_type}',
                    mu=lb_interval['mu_interval'],
                    input_data=lb_data,
                    parameters=parameters,
                    model=mr_model,
                    root_area=reference_area,
                    root_sex=reference_sex,
                    root_year=reference_year,
                    zero_re=zero_re
                )
                lb.update(lb_cov)
            else:
                lb['pi'] = lb_interval['mu_interval']

            lb_disp = covariates.dispersion_covariate_model(
                data_type=f'lb_{_data_type}',
                input_data=lb_data,
                lower=1e12,
                upper=1e13
            )
            lb.update(lb_disp)

            se_lb = lb_data['standard_error'].mask(
                lb_data['standard_error'].le(0) | lb_data['standard_error'].isna(),
                (lb_data['upper_ci'] - lb_data['lower_ci']) / (2 * 1.96)
            )
            ess_lb = lb_data['effective_sample_size'].fillna(
                lb_data['value'] * (1 - lb_data['value']) / se_lb**2
            )
            lb_data['standard_error'] = se_lb
            lb_data['effective_sample_size'] = ess_lb

            lb_like = likelihood.neg_binom_lower_bound(
                name=f'lb_{_data_type}',
                pi=lb_interval['pi'],
                delta=lb_interval['delta'],
                p=lb_data['value'].to_numpy(),
                n=lb_data['effective_sample_size'].to_numpy().astype(int)
            )
            lb.update(lb_like)
            vars['lb'] = lb

    result[data_type] = vars
    return pm_model, result




def consistent(
    model,
    reference_area='all',
    reference_sex='total',
    reference_year='all',
    priors=None,
    zero_re=True,
    rate_type='neg_binom'
):
    """
    Build a consistent multi-rate disease natural history model.
    """
    if priors is None:
        priors = {}

    # 1) Inject any user-specified effect priors
    for t in ['i', 'r', 'f', 'p', 'pf', 'rr']:
        if t in priors:
            model.parameters[t].setdefault('random_effects', {})
            model.parameters[t]['random_effects'].update(priors[t].get('random_effects', {}))
            model.parameters[t].setdefault('fixed_effects', {})
            model.parameters[t]['fixed_effects'].update(priors[t].get('fixed_effects', {}))

    # 2) Normalize rate_type to dict for all keys
    keys = ['i','r','f','p','pf','m_wo','rr','smr','m_with','X']
    if isinstance(rate_type, str):
        rt_map = {k: rate_type for k in keys}
    else:
        rt_map = {k: rate_type.get(k, rate_type) for k in keys}

    rate = {}
    ages = np.array(model.parameters['ages'], dtype=float)

    with pm.Model() as m:
        # 3) Fit incidence, remission, fatal
        for t in ['i','r','f']:
            asr_vars = age_specific_rate(
                model, t,
                reference_area, reference_sex, reference_year,
                mu_age=None,
                mu_age_parent=priors.get((t,'mu')),
                sigma_age_parent=priors.get((t,'sigma')),
                rate_type=rt_map[t],
                zero_re=zero_re
            )
            rate[t] = asr_vars[t]

            # Initialize spline-knots from data/prior
            df = model.get_data(t)
            init = (np.array(priors[t].get('mu'))
                    if t in priors and isinstance(priors[t], dict)
                    else rate[t]['mu_age'].initval.copy())
            if not df.empty:
                mean_df = df.groupby(['age_start','age_end']).mean().reset_index()
                for _, row in mean_df.iterrows():
                    start = int(row['age_start'] - rate[t]['ages'][0])
                    end   = int(row['age_end']   - rate[t]['ages'][0])
                    init[start:end+1] = row['value']
            for i_k, knot in enumerate(rate[t]['knots']):
                rate[t]['gamma'][i_k].initval = np.log(init[int(knot - rate[t]['ages'][0])] + 1e-9)

        # 4) Prepare all-cause mortality m_all
        df_all = model.get_data('m_all')
        if df_all.empty:
            m_all = np.full_like(ages, 0.01)
        else:
            mean_m = df_all.groupby(['age_start','age_end']).mean().reset_index()
            ks = [(r['age_start']+r['age_end']+1)/2. for _,r in mean_m.iterrows()]
            vs = mean_m['value'].tolist()
            knots = [ages[0]] + ks + [ages[-1]]
            vals  = [vs[0]] + vs + [vs[-1]]
            m_all = scipy.interpolate.interp1d(knots, vals, kind='linear')(ages)

        # 5) Prevalence p via ODE / analytic
        logit_C0 = pm.Uniform('logit_C0', lower=-15, upper=15, initval=-10)

        def _compute_p(logit_C0, i_vals, r_vals, f_vals):
            # analytic when remission large
            if r_vals.min() > 5.99:
                return i_vals / (r_vals + m_all + f_vals)
            C0 = pm.math.invlogit(logit_C0)
            S = np.zeros_like(ages)
            C = np.zeros_like(ages)
            dismod_mr.model.ode.ode_function(
                susceptible=S,
                condition=C,
                num_step=2,
                age_local=ages,
                all_cause=m_all,
                incidence=i_vals,
                remission=r_vals,
                excess=f_vals,
                s0=1 - C0,
                c0=C0
            )
            p = C / (S + C)
            return pm.math.where(pm.math.isnan(p), 0.0, p)

        mu_age_p = pm.Deterministic(
            'mu_age_p',
            _compute_p(
                logit_C0,
                rate['i']['mu_age'],
                rate['r']['mu_age'],
                rate['f']['mu_age']
            )
        )

        rate['p'] = age_specific_rate(
            model, 'p',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_p,
            mu_age_parent=priors.get(('p','mu')),
            sigma_age_parent=priors.get(('p','sigma')),
            rate_type=rt_map['p'],
            include_covariates=False,
            zero_re=zero_re
        )['p']

        # 6) Prevalence–fatal pf
        mu_age_pf = pm.Deterministic(
            'mu_age_pf',
            rate['p']['mu_age'] * rate['f']['mu_age']
        )
        rate['pf'] = age_specific_rate(
            model, 'pf',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_pf,
            mu_age_parent=priors.get(('pf','mu')),
            sigma_age_parent=priors.get(('pf','sigma')),
            lower_bound='csmr',
            include_covariates=False,
            zero_re=zero_re
        )['pf']

        # 7) Non‐fatal mortality m_wo
        mu_age_m = pm.Deterministic(
            'mu_age_m',
            pm.math.clip(m_all - rate['pf']['mu_age'], 1e-6, 1e6)
        )
        rate['m_wo'] = age_specific_rate(
            model, 'm_wo',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_m,
            include_covariates=False,
            zero_re=zero_re
        )['m_wo']

        # 8) Relative risk rr
        mu_age_rr = pm.Deterministic(
            'mu_age_rr',
            (rate['m_wo']['mu_age'] + rate['f']['mu_age']) / rate['m_wo']['mu_age']
        )
        rate['rr'] = age_specific_rate(
            model, 'rr',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_rr,
            rate_type=rt_map['rr'],
            include_covariates=False,
            zero_re=zero_re
        )['rr']

        # 9) Standardized mortality ratio smr
        mu_age_smr = pm.Deterministic(
            'mu_age_smr',
            (rate['m_wo']['mu_age'] + rate['f']['mu_age']) / m_all
        )
        rate['smr'] = age_specific_rate(
            model, 'smr',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_smr,
            rate_type=rt_map['smr'],
            include_covariates=False,
            zero_re=zero_re
        )['smr']

        # 10) With‐mortality m_with
        mu_age_m_with = pm.Deterministic(
            'mu_age_m_with',
            rate['m_wo']['mu_age'] + rate['f']['mu_age']
        )
        rate['m_with'] = age_specific_rate(
            model, 'm_with',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_m_with,
            rate_type=rt_map['m_with'],
            include_covariates=False,
            zero_re=zero_re
        )['m_with']

        # 11) Duration X
        def _compute_X(r_vals, m_vals, f_vals):
            hazard = r_vals + m_vals + f_vals
            pr_not = pm.math.exp(-hazard)
            # manual reverse‐loop
            X = np.zeros_like(hazard)
            X[-1] = 1.0 / hazard[-1]
            for idx in range(len(hazard)-2, -1, -1):
                X[idx] = (
                    pr_not[idx] * (X[idx+1] + 1) +
                    (1 - pr_not[idx]) / hazard[idx] -
                    pr_not[idx]
                )
            return X

        mu_age_X = pm.Deterministic(
            'mu_age_X',
            _compute_X(
                rate['r']['mu_age'],
                rate['m_wo']['mu_age'],
                rate['f']['mu_age']
            )
        )
        rate['X'] = age_specific_rate(
            model, 'X',
            reference_area, reference_sex, reference_year,
            mu_age=mu_age_X,
            rate_type=rt_map['X'],
            include_covariates=True,
            zero_re=zero_re
        )['X']

    
    return rate
