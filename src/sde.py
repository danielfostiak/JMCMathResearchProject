"""Scalar stochastic differential equations.

A small toolkit for the finite-dimensional theory of Section 4: Brownian
motion, the Euler-Maruyama scheme, two test equations with known solutions,
and strong/weak convergence diagnostics.

Conventions used throughout:

    time grid          t : (N+1,) array, t[0] = 0, t[N] = T
    paths              W, X : (M, N+1) arrays, column 0 equal to 0 or X0
    drift, diffusion   f(t, x), g(t, x) : callables vectorised over x
    randomness         pass rng = np.random.default_rng(seed) for reproducibility
"""

import numpy as np
import matplotlib.pyplot as plt


def make_grid(T, N):
    """Uniform time grid on [0, T] with N+1 nodes."""
    return np.linspace(0.0, T, N + 1)


def simulate_bm(T, N, M=1, rng=None):
    """Simulate M independent scalar Brownian paths on [0, T] with N steps.

    Returns the time grid t (N+1,) and paths W (M, N+1) with W[:, 0] = 0.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = make_grid(T, N)
    dt = T / N
    dW = rng.normal(0.0, np.sqrt(dt), size=(M, N))
    W = np.concatenate([np.zeros((M, 1)), np.cumsum(dW, axis=1)], axis=1)
    return t, W


def euler_maruyama(f, g, X0, T, N, M=1, rng=None, return_dW=False):
    """Euler-Maruyama solution of dX = f(t, X) dt + g(t, X) dW, X(0) = X0.

    f and g are vectorised callables (t, x) -> (M,) array; X0 is a scalar or
    an (M,) array of initial values. Returns the grid t (N+1,) and the paths
    X (M, N+1); when return_dW is True the Brownian increments dW (M, N) are
    returned as well, so an exact solution can be evaluated on the same path.
    """
    rng = np.random.default_rng() if rng is None else rng
    t = make_grid(T, N)
    dt = T / N
    X = np.zeros((M, N + 1))
    X[:, 0] = X0
    dW = rng.normal(0.0, np.sqrt(dt), size=(M, N))
    for n in range(N):
        X[:, n + 1] = X[:, n] + f(t[n], X[:, n]) * dt + g(t[n], X[:, n]) * dW[:, n]
    return (t, X, dW) if return_dW else (t, X)


# --- Test equations with known solutions, used as benchmarks ---------------

def ou(theta, mu, sigma):
    """Ornstein-Uhlenbeck coefficients for dX = theta (mu - X) dt + sigma dW."""
    return (lambda t, x: theta * (mu - x),
            lambda t, x: sigma * np.ones_like(x))


def gbm(mu, sigma):
    """Geometric Brownian motion coefficients for dX = mu X dt + sigma X dW."""
    return (lambda t, x: mu * x,
            lambda t, x: sigma * x)


def gbm_exact(X0, mu, sigma, t, W):
    """Closed-form GBM path X_t = X0 exp((mu - sigma^2 / 2) t + sigma W_t),
    evaluated on the same Brownian path W that drives the simulation."""
    return X0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


# --- Convergence diagnostics -----------------------------------------------

def _bm_from_increments(dW):
    """Reconstruct Brownian paths (M, N+1) from their increments dW (M, N)."""
    M = dW.shape[0]
    return np.concatenate([np.zeros((M, 1)), np.cumsum(dW, axis=1)], axis=1)


def strong_error(scheme, exact, T, N, M, rng=None, p=2):
    """Strong L^p error at time T: the path-wise distance between a scheme and
    the exact solution driven by the same Brownian path."""
    rng = np.random.default_rng() if rng is None else rng
    t, X, dW = scheme(T, N, M, rng)
    X_ref = exact(t, _bm_from_increments(dW))
    diff = np.abs(X[:, -1] - X_ref[:, -1])
    return float(np.mean(diff**p) ** (1.0 / p))


def weak_error(scheme, exact, phi, T, N, M, rng=None):
    """Weak error |E[phi(X_T)] - E[phi(X_T^exact)]|: distance in distribution
    measured through the test function phi."""
    rng = np.random.default_rng() if rng is None else rng
    t, X, dW = scheme(T, N, M, rng)
    X_ref = exact(t, _bm_from_increments(dW))
    return float(abs(np.mean(phi(X[:, -1])) - np.mean(phi(X_ref[:, -1]))))


def convergence_study(error_fn, Ns, T):
    """Evaluate error_fn(N) over a range of step counts and fit a power law.

    Returns the step sizes dt, the errors, and the fitted log-log slope and
    intercept; the slope estimates the order of convergence.
    """
    Ns = np.asarray(list(Ns))
    dts = T / Ns
    errs = np.array([error_fn(int(N)) for N in Ns])
    slope, intercept = np.polyfit(np.log(dts), np.log(errs), 1)
    return dts, errs, float(slope), float(intercept)


def loglog_convergence(dts, errs, slope, label, ref_order, ax=None):
    """Log-log plot of error against dt with a reference line of given order."""
    ax = plt.gca() if ax is None else ax
    ax.loglog(dts, errs, "o-", label=f"{label}  (fitted slope {slope:.2f})")
    c = errs[-1] / dts[-1] ** ref_order            # anchor the guide at the finest dt
    ax.loglog(dts, c * dts ** ref_order, "k--", alpha=0.6,
              label=fr"reference slope {ref_order:g}")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax
