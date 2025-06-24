import numpy as np
import pymc as pm
import dismod_mr_pymc5
import dismod_mr
from dismod_mr.model import spline, priors, likelihood, covariates, age_groups


def get_default_age_weights():
    """
    Get default age weights for age standardization.
    These represent standard population weights for ages 0-100.
    """
    return [1.9978, 1.9978, 1.9978, 
            1.9316479999999998, 1.9316479999999998, 1.9316479999999998, 1.9316479999999998, 1.9316479999999998, 
            1.7987220000000002, 1.7987220000000002, 1.7987220000000002, 1.7987220000000002, 1.7987220000000002, 
            1.657826, 1.657826, 1.657826, 1.657826, 1.657826, 
            1.560244, 1.560244, 1.560244, 1.560244, 1.560244, 
            1.518288, 1.518288, 1.518288, 1.518288, 1.518288, 
            1.464342, 1.464342, 1.464342, 1.464342, 1.464342, 
            1.36561, 1.36561, 1.36561, 1.36561, 1.36561, 
            1.22947, 1.22947, 1.22947, 1.22947, 1.22947, 
            1.102266, 1.102266, 1.102266, 1.102266, 1.102266, 
            0.982624, 0.982624, 0.982624, 0.982624, 0.982624, 
            0.8691720000000001, 0.8691720000000001, 0.8691720000000001, 0.8691720000000001, 0.8691720000000001, 
            0.736446, 0.736446, 0.736446, 0.736446, 0.736446, 
            0.597018, 0.597018, 0.597018, 0.597018, 0.597018, 
            0.453052, 0.453052, 0.453052, 0.453052, 0.453052, 
            0.319516, 0.319516, 0.319516, 0.319516, 0.319516, 
            0.21945800000000001, 0.21945800000000001, 0.21945800000000001, 0.21945800000000001, 0.21945800000000001, 
            0.1209038, 0.1209038, 0.1209038, 0.1209038, 0.1209038, 
            0.0493326, 0.0493326, 0.0493326, 0.0493326, 0.0493326]


def age_specific_rate(
    mr_model: dismod_mr.data.MRModel,
    data_type,
    reference_area='Global',
    reference_sex='Both',
    reference_year='all',
    mu_age=None,
    mu_age_parent=None,
    sigma_age_parent=None,
    rate_type='neg_binom',
    lower_bound=None,
    interpolation_method='linear',
    include_covariates=True,
    zero_re=False,
    age_weights=None
):
    """
    Generate PyMC objects for epidemiological age-interval data model.
    
    Parameters
    ----------
    mr_model : dismod_mr.data.MRModel
        The model containing data and parameters
    data_type : str
        Type of epidemiological data ('i', 'r', 'f', 'p', etc.)
    reference_area : str, optional
        Reference area for covariate modeling, default 'Global'
    reference_sex : str, optional
        Reference sex for covariate modeling, default 'Both'
    reference_year : str, optional
        Reference year for covariate modeling, default 'all'
    mu_age : array-like, optional
        Prior mean for age-specific rates
    mu_age_parent : array-like, optional
        Parent prior mean for age-specific rates
    sigma_age_parent : array-like, optional
        Parent prior standard deviation for age-specific rates
    rate_type : str, optional
        Likelihood model type, default 'neg_binom'
    lower_bound : str, optional
        Lower bound data type for bounded models
    interpolation_method : str, optional
        Interpolation method for age standardization, default 'linear'
    include_covariates : bool, optional
        Whether to include covariate effects, default True
    zero_re : bool, optional
        Whether to set random effects to zero, default False
    age_weights : array-like, optional
        Age weights for age standardization. If None, uses default weights.
    """
    
    # Use default age weights if none provided
    if age_weights is None:
        age_weights = get_default_age_weights()
    
    age_weights = np.array(age_weights)
    
    _data_type = data_type
    result = {}

    # 0) ignore NaN in parent prior
    if isinstance(mu_age_parent, np.ndarray) and np.isnan(mu_age_parent).any() or \
       isinstance(sigma_age_parent, np.ndarray) and np.isnan(sigma_age_parent).any():
        mu_age_parent = None
        sigma_age_parent = None

    # 0) fetch ages from parameters, then index parameters by data_type.
    ages = np.array(mr_model.parameters['ages'])
    data = mr_model.get_data(data_type)
    lb_data = mr_model.get_data(lower_bound) if lower_bound else None
    parameters = mr_model.parameters.get(data_type, {}) # parameter no longer has other data types and 'ages'

    var_dict = {} # used to be vars, initialized as 'vars = ModelVars()'
    var_dict['data'] = data

    # 1) spline knots & smoothing
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

    # ------------------ Start of PyMC model ------------------
    # with pm.Model(coords=coords) as model:
    with pm.Model() as model:

        # 2) spline prior (spline.py)
        if mu_age is None:
            spline_vars = spline.spline( # uses pm.Normal, pm.Potential, pm.Deterministic
                data_type=_data_type,
                ages=ages,
                knots=knots,
                smoothing=smoothing,
                interpolation_method=interpolation_method
            )
            # returns {"gamma": gamma, "mu_age": mu_age, "ages": ages, "knots": knots}: in spline.py
            var_dict['gamma'] = spline_vars['gamma']
            var_dict['mu_age'] = spline_vars['mu_age']
            var_dict['ages'] = spline_vars['ages']
            var_dict['knots'] = spline_vars['knots']
        else:
            var_dict['mu_age'] = mu_age
            var_dict['ages'] = ages

        # 3) level & derivative constraints (priors.py)
        lc = priors.level_constraints( # uses pm.Normal, pm.Potential and more
            data_type=_data_type,
            parameters=parameters,
            unconstrained_mu_age=var_dict['mu_age'],
            ages=ages                                
        )
        var_dict["mu_age"] = lc["mu_age"]
        var_dict["unconstrained_mu_age"] = lc["unconstrained_mu_age"]
        var_dict["mu_sim"] = lc["mu_sim"]

        dc = priors.derivative_constraints( # uses pm.Normal, pm.Potential and more
            data_type=_data_type,
            parameters=parameters,
            mu_age=var_dict['mu_age'],
            ages=ages
        )
        var_dict["mu_age_derivative_potential"] = dc["mu_age_derivative_potential"]

        # 4) hierarchical similarity (priors.py)
        if mu_age_parent is not None:
            sim = priors.similar( # uses pm.Normal, pm.Potential and more
                data_type=_data_type,
                mu_child=vars['mu_age'],
                mu_parent=mu_age_parent,
                sigma_parent=sigma_age_parent,
                sigma_difference=0.0
            )
            var_dict["parent_similarity"] = sim["parent_similarity"]


        # 5) age‐interval average (age_groups.py)
        if len(data) > 0: # data is a dataframe filtered by data_type
            data = data.copy()

            # 5-1) TODO: recalculating?? standard_error, effective_sample_size 
            se = data['standard_error'].mask(
                data['standard_error'] < 0,
                (data['upper_ci'] - data['lower_ci']) / (2 * 1.96)
            )
            ess = data['effective_sample_size'].fillna(
                data['value'] * (1 - data['value']) / se**2
            )
            data['standard_error'] = se
            data['effective_sample_size'] = ess
          
            age_int = age_groups.age_standardize_approx( # uses pm.Deterministic
                name=_data_type,
                age_weights=age_weights,
                mu_age=var_dict['mu_age'],
                age_start=data['age_start'],
                age_end=data['age_end'],
                ages=ages
            )
            var_dict["mu_interval"] = age_int["mu_interval"]

            # 5-2) covariate & pi (covariates.py)
            if include_covariates: 
                cov = covariates.mean_covariate_model( # uses pm.Normal, pm.Potential and more
                    data_type=_data_type,
                    mu=var_dict['mu_interval'],
                    input_data=data,
                    parameters=parameters,
                    model=mr_model,
                    root_area=reference_area,
                    root_sex=reference_sex,
                    root_year=reference_year,
                    zero_re=zero_re
                )
                var_dict['pi'] = cov['pi']
                var_dict['U'] = cov['U']
                var_dict['U_shift'] = cov['U_shift']
                var_dict['sigma_alpha'] = cov['sigma_alpha']
                var_dict['alpha'] = cov['alpha']
                var_dict['alpha_potentials'] = cov['alpha_potentials']
                var_dict['const_alpha_sigma'] = cov['const_alpha_sigma']
                var_dict['X'] = cov['X']
                var_dict['X_shift'] = cov['X_shift']
                var_dict['beta'] = cov['beta']
                var_dict['const_beta_sigma'] = cov['const_beta_sigma']
                var_dict['hierarchy'] = cov['hierarchy']
            else:
                var_dict['pi'] = var_dict['mu_interval']

        else:
            # 데이터가 없으면 covariate 모델만
            if include_covariates:
                cov = covariates.mean_covariate_model( # uses pm.Normal, pm.Potential and more
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
                var_dict['pi'] = cov['pi']
                var_dict['U'] = cov['U']
                var_dict['U_shift'] = cov['U_shift']
                var_dict['sigma_alpha'] = cov['sigma_alpha']
                var_dict['alpha'] = cov['alpha']
                var_dict['alpha_potentials'] = cov['alpha_potentials']
                var_dict['const_alpha_sigma'] = cov['const_alpha_sigma']
                var_dict['X'] = cov['X']
                var_dict['X_shift'] = cov['X_shift']
                var_dict['beta'] = cov['beta']
                var_dict['const_beta_sigma'] = cov['const_beta_sigma']
                var_dict['hierarchy'] = cov['hierarchy']

        # 6) RATE_TYPE별 전처리 + likelihood 호출 (covariates.py, likelihood.py)
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

                disp_cov = covariates.dispersion_covariate_model( # uses pm.Normal, pm.Potential and more
                    data_type=_data_type,
                    input_data=data,
                    delta_lb=lower,
                    delta_ub=lower * 9.0
                )
                var_dict['eta'] = disp_cov['eta']
                var_dict['Z'] = disp_cov['Z']
                var_dict['zeta'] = disp_cov['zeta']
                var_dict['delta'] = disp_cov['delta']

                nb_dict = likelihood.neg_binom( # uses pm.Deterministic, pm.Potential and more
                    name=_data_type,
                    pi=var_dict['pi'],
                    delta=var_dict['delta'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                var_dict['p_obs'] = nb_dict['p_obs']
                var_dict['p_pred'] = nb_dict['p_pred']

            elif rate_type == 'log_normal':
                missing = data['standard_error'] < 0
                if missing.any():
                    data.loc[missing, 'standard_error'] = 1e6

                var_dict['sigma'] = pm.Uniform(
                    name=f'sigma_{_data_type}',
                    lower=1e-4,
                    upper=1.0,
                    initval=1e-2
                )

                lg_dict = likelihood.log_normal(
                    data_type=_data_type,
                    pi=var_dict['pi'],
                    sigma=var_dict['sigma'],
                    p=data['value'].to_numpy(),
                    s=data['standard_error'].to_numpy()
                )
                var_dict['p_obs'] = lg_dict['p_obs']
                var_dict['p_pred'] = lg_dict['p_pred']

            elif rate_type == 'normal':
                missing = data['standard_error'] < 0
                if missing.any():
                    data.loc[missing, 'standard_error'] = 1e6

                var_dict['sigma'] = pm.Uniform(
                    name=f'sigma_{_data_type}',
                    lower=1e-4,
                    upper=1e-1,
                    initval=1e-2
                )

                nm_dict = likelihood.normal(
                    name=_data_type,
                    pi=var_dict['pi'],
                    sigma=var_dict['sigma'],
                    p=data['value'].to_numpy(),
                    s=data['standard_error'].to_numpy()
                )
                var_dict['p_obs'] = nm_dict['p_obs']
                var_dict['p_pred'] = nm_dict['p_pred']

            elif rate_type == 'binom':
                bad_ess = data['effective_sample_size'] < 0
                if bad_ess.any():
                    data.loc[bad_ess, 'effective_sample_size'] = 0.0

                bb_dict = likelihood.binom(
                    name=_data_type,
                    pi=var_dict['pi'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                var_dict['p_obs'] = bb_dict['p_obs']
                var_dict['p_pred'] = bb_dict['p_pred']

            elif rate_type == 'beta_binom':
                bbb_dict = likelihood.beta_binom(
                    name=_data_type,
                    pi=var_dict['pi'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                var_dict['p_obs'] = bbb_dict['p_obs']
                var_dict['p_pred'] = bbb_dict['p_pred']

            elif rate_type == 'poisson':
                bad_ess = data['effective_sample_size'] < 0
                if bad_ess.any():
                    data.loc[bad_ess, 'effective_sample_size'] = 0.0

                pois_dict = likelihood.poisson(
                    name=_data_type,
                    pi=var_dict['pi'],
                    p=data['value'].to_numpy(),
                    n=data['effective_sample_size'].to_numpy().astype(int)
                )
                var_dict['p_obs'] = pois_dict['p_obs']
                var_dict['p_pred'] = pois_dict['p_pred']

            elif rate_type == 'offset_log_normal':
                var_dict['sigma'] = pm.Uniform(
                    name=f'sigma_{_data_type}',
                    lower=1e-4,
                    upper=10.0,
                    initval=1e-2
                )

                oln_dict = likelihood.offset_log_normal(
                    name=_data_type,
                    pi=var_dict['pi'],
                    sigma=var_dict['sigma'],
                    p=data['value'].to_numpy(),
                    s=data['standard_error'].to_numpy()
                )
                var_dict['p_obs_potential'] = oln_dict['p_obs_potential']
                var_dict['p_pred'] = oln_dict['p_pred']
                var_dict['p_zeta'] = oln_dict['p_zeta']

            else:
                raise ValueError(f'Unsupported rate_type "{rate_type}"')
            
        # else:
            # 6) 데이터가 전혀 없으면 likelihood 호출 없음

        # 7) covariate‐level constraints
        if include_covariates:
            clc = priors.covariate_level_constraints(
                data_type=_data_type,
                model=mr_model,
                vars=var_dict,
                ages=ages
            )
            var_dict["covariate_constraint"] = clc["covariate_constraint"]

        # 8) lower‐bound 데이터 처리 (옵션)
        if lb_data is not None and len(lb_data) > 0:
            lb = {}
            lb_interval = age_groups.age_standardize_approx(
                name=f'lb_{_data_type}',
                age_weights=age_weights,
                mu_age=var_dict['mu_age'],
                age_start=lb_data['age_start'],
                age_end=lb_data['age_end'],
                ages=ages
            )
            lb["mu_interval"] = lb_interval["mu_interval"]

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
                lb['pi'] = lb_cov['pi']
                lb['U'] = lb_cov['U']
                lb['U_shift'] = lb_cov['U_shift']
                lb['sigma_alpha'] = lb_cov['sigma_alpha']
                lb['alpha'] = lb_cov['alpha']
                lb['alpha_potentials'] = lb_cov['alpha_potentials']
                lb['const_alpha_sigma'] = lb_cov['const_alpha_sigma']
                lb['X'] = lb_cov['X']
                lb['X_shift'] = lb_cov['X_shift']
            else:
                lb['pi'] = lb['mu_interval']

            lb_disp = covariates.dispersion_covariate_model(
                data_type=f'lb_{_data_type}',
                input_data=lb_data,
                lower=1e12,
                upper=1e13
            )
            lb['eta'] = lb_disp['eta']
            lb['Z'] = lb_disp['Z']
            lb['zeta'] = lb_disp['zeta']
            lb['delta'] = lb_disp['delta']

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
            lb['p_obs'] = lb_like['p_obs']
            var_dict['lb'] = lb
            

        result[data_type] = var_dict
        return model, result
