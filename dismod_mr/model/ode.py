import numpy as np
from numba import njit

@njit
def f(
    a: float,
    susceptible_condition: np.ndarray,
    incidence: np.ndarray,
    remission: np.ndarray,
    excess: np.ndarray,
    all_cause: np.ndarray
) -> np.ndarray:
    """
    RHS of the disease ODE system for one age point.

    Parameters
    ----------
    a : float
        Current age (offset handled externally).
    susceptible_condition : array-like, shape (2,)
        [S, C] at this age.
    incidence : array-like
        Incidence rates by age index.
    remission : array-like
        Remission rates by age index.
    excess : array-like
        Excess mortality rates.
    all_cause : array-like
        All-cause mortality rates.

    Returns
    -------
    out : ndarray, shape (2,)
        [dS/da, dC/da]
    """
    s = susceptible_condition[0]
    c = susceptible_condition[1]
    i = incidence[int(a)]
    r = remission[int(a)]
    e = excess[int(a)]
    m = all_cause[int(a)]
    other = m - e * s / (s + c)
    ds_da = - (i + other) * s + r * c
    dc_da = i * s - (r + other + e) * c
    return np.array([ds_da, dc_da])

@njit
def ode_function(
    susceptible: np.ndarray,
    condition: np.ndarray,
    num_step: int,
    age_local: np.ndarray,
    all_cause: np.ndarray,
    incidence: np.ndarray,
    remission: np.ndarray,
    excess: np.ndarray,
    s0: float,
    c0: float
) -> None:
    """
    Solve the age-structured ODE system using RK4 steps in-place.

    Parameters
    ----------
    susceptible : ndarray, shape (N,)
        Pre-allocated array to fill S(a).
    condition : ndarray, shape (N,)
        Pre-allocated array to fill C(a).
    num_step : int
        Number of sub-steps per age interval.
    age_local : ndarray, shape (N,)
        Age grid points (monotonic).
    all_cause, incidence, remission, excess : ndarray
        Rate arrays by age index.
    s0, c0 : float
        Initial conditions at age_local[0].

    Modifies
    --------
    susceptible, condition arrays in-place.
    """
    N = age_local.shape[0]
    t0 = age_local[0]
    susceptible[0] = s0
    condition[0]   = c0
    sc = np.array([s0, c0])
    for j in range(N - 1):
        a_step = (age_local[j+1] - age_local[j]) / num_step
        ti = age_local[j]
        yi = sc.copy()
        for _ in range(num_step):
            k1 = a_step * f(ti - t0, yi, incidence, remission, excess, all_cause)
            k2 = a_step * f(ti - t0 + 0.5*a_step, yi + 0.5*k1, incidence, remission, excess, all_cause)
            k3 = a_step * f(ti - t0 + 0.5*a_step, yi + 0.5*k2, incidence, remission, excess, all_cause)
            k4 = a_step * f(ti - t0 + a_step, yi + k3, incidence, remission, excess, all_cause)
            yi = yi + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            ti += a_step
        susceptible[j+1] = yi[0]
        condition[j+1]   = yi[1]
        sc = yi
