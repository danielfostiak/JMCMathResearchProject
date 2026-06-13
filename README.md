# Cylindrical Wiener Process and Stochastic Heat Equation

Computational component of our second-year group research project (Imperial
College London, Department of Mathematics). The report develops the theory of
cylindrical Wiener processes and applies it to the stochastic heat equation;
this repository contains the simulations and the figures that accompany it.

**Authors:** Git Lun Mak, Nathan Gonzalez, Daniel Fostiak, Yuvraj Singh
&nbsp;·&nbsp; **Supervisor:** Tomasz Kosmala

## Contents

| Path | Description |
| --- | --- |
| `src/sde.py` | Scalar SDE toolkit: Brownian motion, Euler–Maruyama, the OU and GBM test equations, and strong/weak convergence diagnostics. |
| `src/heat.py` | Stochastic heat equation solvers: an explicit finite-difference scheme and the spectral (sine-basis) method. |
| `notebooks/01_sde_foundations.ipynb` | Brownian motion and the test SDEs (report §3–4). |
| `notebooks/02_em_convergence.ipynb` | Strong and weak convergence of Euler–Maruyama (report §4.1, §12). |
| `notebooks/03_stochastic_heat.ipynb` | Simulation of the stochastic heat equation (report §11). |
| `figures/` | Figures produced by the notebooks. |

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Each notebook imports the solvers from `src/` and regenerates its figures into
`figures/`; run them top to bottom.
