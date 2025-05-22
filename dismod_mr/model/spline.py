import numpy as np
import pymc as pm
import scipy.interpolate
import aesara.tensor as at
from aesara.compile.ops import as_op


def spline(data_type, ages, knots, smoothing, interpolation_method='linear'):
    """
    PyMC5 spline model: supports multiple interpolation methods via SciPy wrapped in Aesara.
    Input/output signature unchanged from PyMC2/3.

    Parameters
    ----------
    data_type : str
    ages : array-like (N,) points to interpolate to
    knots : array-like (K,) strictly increasing knot locations
    smoothing : float smoothing parameter (0 = none, inf = none)
    interpolation_method : str, one of 'linear','nearest','zero','slinear','quadratic','cubic'

    Returns
    -------
    dict with keys:
      'gamma': list of gamma_k RVs,
      'mu_age': Deterministic tensor of shape (N,),
      'ages': input ages array,
      'knots': input knots array
      'smooth_gamma' (optional): smoothing potential added inside model
    """
    ages = np.asarray(ages)
    knots = np.asarray(knots)
    assert np.all(np.diff(knots) > 0), "Spline knots must be strictly increasing"

    with pm.Model() as spline_model:
        K = len(knots)
        # gamma vector: log rates at knots
        gamma_vec = pm.Normal(
            f"gamma_{data_type}", mu=0.0, sigma=10.0,
            initval=np.full(K, -10.0), shape=(K,)
        )
        gamma = [gamma_vec[i] for i in range(K)]

        # wrap SciPy interp in Aesara op
        @as_op(itypes=[at.dvector], otypes=[at.dvector])
        def interp_op(g_vals):
            # g_vals: log-rate at knots
            f = scipy.interpolate.interp1d(
                knots,
                np.exp(g_vals),
                kind=interpolation_method,
                bounds_error=False,
                fill_value=0.0
            )
            return f(ages).astype(np.float64)

        mu_age = pm.Deterministic(
            f"mu_age_{data_type}", interp_op(gamma_vec)
        )

        # optional smoothing prior
        if (smoothing > 0) and np.isfinite(smoothing):
            dg = gamma_vec[1:] - gamma_vec[:-1]
            dk = knots[1:] - knots[:-1]
            smooth_val = pm.math.sqrt(pm.math.sum(dg**2 / dk))
            tau = smoothing**-2
            pm.Potential(
                f"smooth_mu_{data_type}",
                -0.5 * tau * smooth_val**2
            )

    return {
        'gamma': gamma,
        'mu_age': mu_age,
        'ages': ages,
        'knots': knots
    }
