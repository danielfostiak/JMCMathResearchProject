"""The stochastic heat equation on (0, 1).

Two solvers for

    du = d u_xx dt + g(t) dW,    u(t, 0) = u(t, 1) = 0,

with W a Q-Wiener process: an explicit finite-difference scheme in physical
space, and the spectral scheme of Section 11 in which the initial profile is
projected onto the sine basis, each modal coefficient is evolved as a
decoupled Ornstein-Uhlenbeck process, and the result is transformed back.
"""

import math

import numpy as np
from scipy.fft import dst, idst


def trace_class_Q(K, scale=10.0):
    """Diagonal trace-class covariance with entries scale / n^2 on K modes
    (the operator Q_2 of Section 11.3)."""
    return np.diag([scale / (n + 1) ** 2 for n in range(K)])


def heat_fd(u0, T, Nx, Nt, rng, k, var):
    """Explicit finite-difference solution in physical space.

    The Laplacian is the standard three-point stencil and an independent
    Gaussian increment is added at each interior node and time step. Returns
    meshgrid arrays X, Y over (time, space) and the solution u (Nt, Nx+1).
    """
    u = np.zeros((Nt, Nx + 1))
    u[0] = [u0(j / Nx) for j in range(Nx + 1)]

    t_grid = np.linspace(0, T, Nt)
    x_grid = np.linspace(0, 1, Nx + 1)
    X, Y = np.meshgrid(t_grid, x_grid)

    dt = T / Nt
    dx = 1 / Nx
    for n in range(1, Nt):
        u[n][0] = 0
        u[n][Nx] = 0
        for j in range(1, Nx):
            noise = rng.normal(0, math.sqrt(var * dt / dx))
            u[n][j] = (u[n - 1][j]
                       + k * (dt / dx**2) * (u[n - 1][j - 1] - 2 * u[n - 1][j] + u[n - 1][j + 1])
                       + noise)
    return X, Y, u


def heat_spectral(u0, T, Nt, Nx, K_modes, rng, d, var, g, Q):
    """Spectral (sine-basis) solution driven by a Q-Wiener process.

    The initial profile is projected onto the first K_modes sine modes with a
    discrete sine transform; each coefficient then follows its Ornstein-
    Uhlenbeck recursion, with noise increments drawn from N(0, var dt Q),
    before the inverse transform maps the state back to physical space.
    Returns meshgrid arrays X, Y over (time, space) and u (Nt, Nx+1).
    """
    u = np.zeros((Nt, Nx + 1))
    u[0] = [u0(j / Nx) for j in range(Nx + 1)]

    t_grid = np.linspace(0, T, Nt)
    x_grid = np.linspace(0, 1, Nx + 1)
    X, Y = np.meshgrid(t_grid, x_grid)

    dt = T / Nt
    a = dst(u[0], n=K_modes, type=1, norm='ortho')
    for n in range(Nt):
        w = rng.multivariate_normal(np.zeros(K_modes), var * dt * Q)
        a = [a[k] * np.exp(-2 * d * ((k + 1) * np.pi) ** 2 * dt) + g(n * dt) * w[k]
             for k in range(K_modes)]
        u[n] = idst(a, n=Nx + 1, type=1, norm='ortho')
    return X, Y, u


def ou_step_variance(K, d, dt, var):
    """Exact one-step Ornstein-Uhlenbeck noise variance for each of K modes (Q = I).

    Integrating the modal SDE exactly over a step of length dt gives mode k the
    increment variance var * (1 - exp(-2 lam_k dt)) / (2 lam_k), with
    lam_k = d (k pi)^2, rather than the flat var * dt used by heat_spectral.
    """
    k = np.arange(1, K + 1)
    lam = d * (k * np.pi) ** 2
    return var * (1.0 - np.exp(-2.0 * lam * dt)) / (2.0 * lam)


def flat_step_variance(K, d, dt, var):
    """Naive flat per-mode variance var * dt (ignores the within-step decay)."""
    return var * dt * np.ones(K)


def heat_spectral_diag(u0, T, Nt, Nx, rng, K_modes, d, g, var_diag):
    """Spectral solver driven by a diagonal per-mode variance vector.

    Like heat_spectral, but the modal noise is independent across modes with the
    variances supplied in var_diag (length K_modes). Innovations are drawn as
    standard normals scaled by sqrt(var_diag), so two variance choices sharing a
    seed are compared on matched noise. Returns X, Y over (time, space) and u.
    """
    dt = T / Nt
    t_grid = np.linspace(0, T, Nt)
    x_grid = np.linspace(0, 1, Nx + 1)
    X, Y = np.meshgrid(t_grid, x_grid)

    k = np.arange(1, K_modes + 1)
    decay = np.exp(-d * (k * np.pi) ** 2 * dt)
    std = np.sqrt(var_diag)

    u = np.zeros((Nt, Nx + 1))
    u[0] = [u0(j / Nx) for j in range(Nx + 1)]
    a = dst(u[0], n=K_modes, type=1, norm='ortho')
    for n in range(1, Nt):
        w = rng.standard_normal(K_modes) * std   # matched innovations under a fixed seed
        a = a * decay + g((n - 1) * dt) * w
        u[n] = idst(a, n=Nx + 1, type=1, norm='ortho')
    return X, Y, u
