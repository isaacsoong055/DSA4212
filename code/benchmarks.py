"""Benchmark landscapes on [-5, 5]^d.

Each Landscape knows its global minimum (x_star, f_star), a success threshold
epsilon on regret, and a global-basin check based on local descent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

LO, HI = -5.0, 5.0

# Primary-landscape defaults (frozen in Phase 1, see docs/pilot_log.md).
PRIMARY = dict(c=0.0, s=10.0, A=4.0, w=0.3)
WELL_LOCATIONS = {"far": (3.0, 3.0), "near": (2.0, 0.0)}

# Validation-function thresholds: set from the epsilon_validation pilot (docs/pilot_log.md).
EPSILON_RASTRIGIN = 0.2  # random N=300 succeeds 5/200; cap (nearest non-global regret) 0.995
EPSILON_SCHWEFEL = 5.0   # random N=300 succeeds 10/200 (exactly 5%); cap 118.4

# Success threshold on regret, chosen from random sampling only (pilot step 2).
EPSILON_PRIMARY = 0.05  # pilot step 2: random N=300 succeeds 5/200 on primary_far_w0.3


@dataclass
class Landscape:
    name: str
    d: int
    func: Callable[[np.ndarray], float]
    x_star: np.ndarray
    f_star: float
    epsilon: float | None = None
    basin_tol: float = 0.1
    meta: dict = field(default_factory=dict)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[LO] * self.d, [HI] * self.d])

    def __call__(self, x) -> float:
        return float(self.func(np.asarray(x, dtype=float)))

    def regret(self, f: float) -> float:
        return f - self.f_star

    def local_descent(self, x) -> np.ndarray:
        """End point of L-BFGS-B started at x, inside the box."""
        res = minimize(self.func, np.asarray(x, float), method="L-BFGS-B",
                       bounds=[(LO, HI)] * self.d)
        return res.x

    def in_global_basin(self, x) -> bool:
        """True if local descent from x reaches the global minimiser."""
        return bool(np.linalg.norm(self.local_descent(x) - self.x_star) < self.basin_tol)

    def is_success(self, best_f: float) -> bool:
        """Primary success: best regret <= epsilon."""
        if self.epsilon is None:
            raise ValueError(f"epsilon not set for {self.name}")
        return self.regret(best_f) <= self.epsilon


def local_minima(ls: Landscape, n_starts: int = 1000, seed: int = 0, merge_tol: float = 1e-2):
    """Distinct local minima (x, regret), sorted by regret, from L-BFGS-B started at
    n_starts uniform points drawn with a fixed seed. A larger n_starts reuses the same first starts."""
    starts = np.random.default_rng(seed).uniform(LO, HI, size=(n_starts, ls.d))
    ends = np.array([ls.local_descent(x) for x in starts])
    fs = np.array([ls(x) for x in ends])
    found: list[tuple[np.ndarray, float]] = []
    for i in np.argsort(fs, kind="stable"):
        if not any(np.linalg.norm(ends[i] - x) < merge_tol for x, _ in found):
            found.append((ends[i], float(fs[i] - ls.f_star)))
    return found


def nearest_nonglobal(ls: Landscape, minima, tol: float = 1e-2):
    """(regret, locations) of the lowest-regret non-global minimum; ties within 1e-6 relative are all returned."""
    others = [(x, r) for x, r in minima if np.linalg.norm(x - ls.x_star) > tol]
    r0 = others[0][1]
    return r0, [x for x, r in others if abs(r - r0) <= 1e-6 * max(1.0, abs(r0))]


def _polish(func, x0, d):
    """Refine an approximate minimiser numerically."""
    res = minimize(func, np.asarray(x0, float), method="L-BFGS-B",
                   bounds=[(LO, HI)] * d, options=dict(ftol=1e-15, gtol=1e-12))
    return res.x, float(res.fun)


def sphere(d: int = 2) -> Landscape:
    return Landscape("sphere", d, lambda x: float(np.sum(x**2)), np.zeros(d), 0.0)


def rastrigin(d: int = 2) -> Landscape:
    def f(x):
        return float(10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
    return Landscape("rastrigin", d, f, np.zeros(d), 0.0, epsilon=EPSILON_RASTRIGIN)


def schwefel(d: int = 2) -> Landscape:
    """Schwefel 2.26 with z = 100 x, so the domain [-500, 500] maps to [-5, 5]."""
    def f(x):
        z = 100.0 * x
        return float(418.9828872724338 * d - np.sum(z * np.sin(np.sqrt(np.abs(z)))))
    x_star, f_star = _polish(f, np.full(d, 4.2096874633), d)
    return Landscape("schwefel", d, f, x_star, f_star, epsilon=EPSILON_SCHWEFEL)


def decoy_well(d: int = 2, w: float = PRIMARY["w"], loc: str = "far",
               A: float = PRIMARY["A"], c: float = PRIMARY["c"],
               s: float = PRIMARY["s"], epsilon: float | None = None) -> Landscape:
    """Broad decoy bowl centred at c plus a narrow Gaussian well at x_g.

    f(x) = ||x - c||^2 / s - A exp(-||x - x_g||^2 / (2 w^2))
    For d > 2 the well location is padded with zeros.
    """
    c_vec = np.full(d, c)
    x_g = np.zeros(d)
    x_g[:2] = WELL_LOCATIONS[loc]

    def f(x):
        return float(np.sum((x - c_vec) ** 2) / s
                     - A * np.exp(-np.sum((x - x_g) ** 2) / (2 * w**2)))

    x_star, f_star = _polish(f, x_g, d)
    name = f"primary_{loc}_w{w:g}"
    eps = EPSILON_PRIMARY if epsilon is None else epsilon
    return Landscape(name, d, f, x_star, f_star, epsilon=eps,
                     basin_tol=min(0.1, w / 2),
                     meta=dict(c=c_vec, x_g=x_g, A=A, s=s, w=w, loc=loc))


def get_landscape(name: str, d: int = 2) -> Landscape:
    """Build a landscape from its id, e.g. 'sphere' or 'primary_far_w0.3'."""
    if name in ("sphere", "rastrigin", "schwefel"):
        return {"sphere": sphere, "rastrigin": rastrigin, "schwefel": schwefel}[name](d)
    m = re.fullmatch(r"primary_(near|far)_w([0-9.]+)", name)
    if m:
        return decoy_well(d, w=float(m.group(2)), loc=m.group(1))
    raise ValueError(f"unknown landscape {name!r}")
