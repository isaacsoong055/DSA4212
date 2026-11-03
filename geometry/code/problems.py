"""Test problems with COCO-style instances.

An *instance* fixes a pseudo-random rotation and a pseudo-random optimum
location, both keyed by an integer instance index.  Following the COCO
guidelines, the optimum is deliberately NOT at the origin and the function is
NOT separable unless we ask for it to be: putting the optimum at all-zeros or
keeping the axes aligned creates an artificial regularity that a solver can
exploit by accident.

Every problem returns a callable plus a dict of metadata so experiment
scripts can record exactly what was optimised.
"""

import numpy as np


def haar_rotation(dimension, rng):
    """Draw a rotation matrix uniformly (Haar measure) via QR of a Gaussian."""
    a = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(a)
    # Fix the signs so Q is uniform on O(n) rather than biased by QR conventions.
    return q @ np.diag(np.sign(np.diag(r)))


def ellipsoid_spectrum(dimension, condition_number):
    """Log-spaced eigenvalues from 1 to condition_number (the classic ellipsoid)."""
    if dimension == 1:
        return np.array([1.0])
    return np.logspace(0.0, np.log10(condition_number), dimension)


def make_ellipsoid(dimension, condition_number, instance=0, rotation="random",
                   optimum_radius=2.0, spectrum=None):
    """Build f(x) = 0.5 (x-x*)^T H (x-x*) with a controlled spectrum and rotation.

    rotation:
        "none"    - H is diagonal, so the problem is separable and axis-aligned.
        "random"  - H = Q D Q^T with Q Haar-random for this instance.
        float     - degrees; a rotation in the (0, 1) plane only.  Use for
                    2-D angle sweeps; 45 degrees in 2-D is a special, maximally
                    adversarial case and should not be the only angle tested.

    The optimum x* is drawn on a sphere of radius optimum_radius, so the search
    never starts at the solution.
    """
    rng = np.random.default_rng(1000 * instance + 7)
    eigenvalues = ellipsoid_spectrum(dimension, condition_number) if spectrum is None \
        else np.asarray(spectrum, dtype=float)

    if rotation == "none":
        q = np.eye(dimension)
    elif rotation == "random":
        q = haar_rotation(dimension, rng)
    else:
        angle = np.deg2rad(float(rotation))
        q = np.eye(dimension)
        q[np.ix_([0, 1], [0, 1])] = [[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle),  np.cos(angle)]]

    hessian = q @ np.diag(eigenvalues) @ q.T

    direction = rng.normal(size=dimension)
    optimum = optimum_radius * direction / np.linalg.norm(direction)

    def objective(x):
        delta = np.asarray(x, dtype=float) - optimum
        return 0.5 * float(delta @ hessian @ delta)

    meta = {"dimension": dimension, "condition_number": condition_number,
            "instance": instance, "rotation": rotation,
            "hessian": hessian, "optimum": optimum,
            "eigenvalues": eigenvalues}
    return objective, meta


def rescale_coordinates(objective, scales):
    """Change of units: x_i is measured in different units than before.

    Returns g(x) = f(S x) with S = diag(scales).  A method invariant to a
    scaling of variables should be unaffected; one that is not will see the
    effective conditioning change.
    """
    scales = np.asarray(scales, dtype=float)

    def rescaled(x):
        return objective(scales * np.asarray(x, dtype=float))
    return rescaled


def translate(objective, shift):
    """Move the optimum: g(x) = f(x - shift)."""
    shift = np.asarray(shift, dtype=float)

    def translated(x):
        return objective(np.asarray(x, dtype=float) - shift)
    return translated


MONOTONE_TRANSFORMS = {
    # Strictly increasing maps of the objective VALUE only; the location of the
    # optimum and the full ordering of candidates are unchanged.  A rank-based
    # method is exactly invariant to these; a method that reads f values is not.
    "identity": lambda v: v,
    "log1p":    lambda v: np.log1p(v),
    "cube":     lambda v: np.sign(v) * np.abs(v) ** 3,
    "sqrt":     lambda v: np.sqrt(np.abs(v)),
    # An additive offset: the optimum's value becomes 1 instead of 0.  COCO
    # gives every function a non-zero f_opt for the same reason.
    "plus1":    lambda v: v + 1.0,
}


def transform_value(objective, name):
    """Compose a strictly monotone transform onto the objective's output."""
    g = MONOTONE_TRANSFORMS[name]

    def transformed(x):
        return float(g(objective(x)))
    return transformed
