import numpy as np
import pymc as pm
import scipy.interpolate
import pytensor.tensor as at
from pytensor.graph.op import Op
from pytensor.tensor.type import TensorType
import pytensor

# Configure PyTensor to minimize graph/dot printing
pytensor.config.optimizer = 'fast_compile'
pytensor.config.profile = False
pytensor.config.profile_optimizer = False
pytensor.config.print_graph = False


class InterpOp(Op):
    """
    A PyTensor Op to interpolate exp(gamma_vec) over ages using SciPy.
    """
    itypes = [TensorType('float64', (False,))]
    otypes = [TensorType('float64', (False,))]

    def __init__(self, knots, ages, method):
        self.knots = np.asarray(knots, dtype=np.float64)
        self.ages = np.asarray(ages, dtype=np.float64)
        self.method = method
        super().__init__()

    def perform(self, node, inputs, outputs):
        (g,) = inputs
        # exponentiate gamma
        exp_g = np.exp(g)
        f = scipy.interpolate.interp1d(
            self.knots, exp_g,
            kind=self.method,
            bounds_error=False,
            fill_value=0.0
        )
        result = f(self.ages).astype(np.float64)
        outputs[0][0] = result

    def infer_shape(self, fgraph, node, input_shapes):
        return [(self.ages.shape[0],)]


def spline(data_type: str,
           ages: np.ndarray,
           knots: np.ndarray,
           smoothing: float,
           interpolation_method: str = 'linear') -> dict:
    """
    Create a spline-based age-specific rate model in PyMC using a custom PyTensor Op.

    Parameters
    ----------
    data_type : str
        Identifier for naming variables.
    ages : array-like
        Array of target ages.
    knots : array-like
        Strictly increasing array of knot locations.
    smoothing : float
        Precision parameter for smoothing prior (0 or inf disables smoothing).
    interpolation_method : str
        One of SciPy interp1d kinds: 'linear', 'slinear', 'quadratic', 'cubic', etc.

    Returns
    -------
    dict
        - 'gamma': list of Normal RVs at each knot
        - 'mu_age': Deterministic RV of interpolated rates at ages
        - 'ages': original ages array
        - 'knots': original knots array
    """
    ages = np.asarray(ages, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    if not np.all(np.diff(knots) > 0):
        raise ValueError('Spline knots must be strictly increasing')

    # Build inside a model context
    with pm.Model() as _model:
        K = len(knots)
        # normal priors on log-scale spline coefficients
        gamma = [
            pm.Normal(f"gamma_{data_type}_{i}", mu=0.0, sigma=10.0, initval=0.0)
            for i in range(K)
        ]
        gamma_vec = at.stack(gamma)

        # interpolation via custom Op
        interp = InterpOp(knots, ages, interpolation_method)
        mu_age = pm.Deterministic(f"mu_age_{data_type}", interp(gamma_vec))

        # optional smoothing prior
        if smoothing > 0 and np.isfinite(smoothing):
            diffs = gamma_vec[1:] - gamma_vec[:-1]
            intervals = knots[1:] - knots[:-1]
            smooth_term = at.sqrt(at.sum(diffs**2 / intervals))
            tau = smoothing**-2
            pm.Potential(f"smooth_{data_type}", -0.5 * tau * smooth_term**2)

    return { 'gamma': gamma, 'mu_age': mu_age, 'ages': ages, 'knots': knots }
