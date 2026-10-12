"""Closed-form reference solutions for the verification tests (numpy and scipy only,
no dolfinx).

Scope: homogeneous states whose gradient F is diagonal in the axes of the frame,
F = diag(l_0, l_1, l_2). The stretches are indexed like the components of the frame:
(x, y, z) for the Cartesian framework, (r, z, theta) for the axisymmetric one. For such
states the first Piola-Kirchhoff stress is diagonal, P_i = d(psi)/d(l_i), and the Cauchy
stress is sigma_i = P_i l_i / J.

The three laws are those of parameters_class.HyperelasticMaterial, with the same
parameter names (a dict read from the JSON file):

    psi = C_10 (I1b - 3) [+ C_01 (I2b - 3), not Neo-Hookean] + (J - 1)^2 / D
          [+ k1 / (2 k2) sum_{i in 4, 6} H(E_i) (exp(k2 E_i^2) - 1), HGO only]
    E_i = kappa (I1b - 3) + (1 - 3 kappa) (I_i - 1),     I_i = l_{axis(a_i)}^2

The fibre directions a4, a6 must lie along one axis of the frame (in the local reference
system of the mesh, which is the frame itself in the verification meshes).

Sign convention of the framework: a follower pressure p > 0 pushes INTO the body, the
traction is -p n, so sigma_nn = -p. A tensile follower load is p < 0.
"""
import numpy as np
from scipy.optimize import fsolve

LAWS = ("Neo-Hookean", "Mooney-Rivlin", "HGO")


# --------------------------------------------------------------------------- parameters

def reference_params(law, frame="cube"):
    """O(1) parameters used by the verification tests, as a dict (JSON-ready).

    frame = "cube": stretches (x, y, z), HGO fibres along x and y.
    frame = "axi" : stretches (r, z, theta), HGO fibres along r and theta.
    """
    if law not in LAWS:
        raise ValueError(f"law must be one of {LAWS}, got {law!r}")
    if frame not in ("cube", "axi"):
        raise ValueError("frame must be 'cube' or 'axi'")
    params = {"sedf_type": law, "C_10": 1.0, "D": 0.5}
    if law in ("Mooney-Rivlin", "HGO"):
        params["C_01"] = -0.2
    if law == "HGO":
        params.update(k1=0.5, k2=2.0, kappa=0.0, a4=[1, 0, 0],
                      a6=[0, 1, 0] if frame == "cube" else [0, 0, 1])
    return params


def _axis(a, name):
    """Index of the frame axis along which the unit-or-not vector a points."""
    a = np.asarray(a, float)
    if a.shape != (3,):
        raise ValueError(f"{name} must have 3 components, got {a}")
    k = int(np.argmax(np.abs(a)))
    if np.any(np.abs(np.delete(a, k)) > 1e-12) or a[k] == 0.0:
        raise ValueError(f"{name} = {a.tolist()} is not along an axis of the frame: "
                         "closed forms exist only for fibres aligned with the stretches")
    return k


def _unpack(params):
    law = params["sedf_type"]
    if law not in LAWS:
        raise ValueError(f"sedf_type must be one of {LAWS}, got {law!r}")
    p = {"law": law, "C10": params["C_10"], "D": params["D"],
         "C01": 0.0 if law == "Neo-Hookean" else params["C_01"]}
    if law == "HGO":
        p.update(k1=params["k1"], k2=params["k2"], kappa=params["kappa"],
                 axes=(_axis(params["a4"], "a4"), _axis(params["a6"], "a6")))
    return p


def _kinematics(lam):
    l = np.asarray(lam, float)
    if l.shape != (3,) or np.any(l <= 0.0):
        raise ValueError(f"stretches must be three positive numbers, got {lam}")
    J = l.prod()
    S1 = np.sum(l ** 2)
    S2 = l[0]**2 * l[1]**2 + l[1]**2 * l[2]**2 + l[0]**2 * l[2]**2
    return l, J, S1, S2


# --------------------------------------------------------------------------- energy, stress

def energy(lam, params):
    """Strain energy density psi(l_0, l_1, l_2)."""
    p = _unpack(params)
    l, J, S1, S2 = _kinematics(lam)
    I1, I2 = J ** (-2 / 3) * S1, J ** (-4 / 3) * S2
    psi = p["C10"] * (I1 - 3) + (J - 1) ** 2 / p["D"]
    if p["law"] != "Neo-Hookean":
        psi += p["C01"] * (I2 - 3)
    if p["law"] == "HGO":
        for ax in p["axes"]:
            E = p["kappa"] * (I1 - 3) + (1 - 3 * p["kappa"]) * (l[ax] ** 2 - 1)
            if E > 0.0:
                psi += p["k1"] / (2 * p["k2"]) * (np.exp(p["k2"] * E ** 2) - 1)
    return psi


def pk1(lam, params):
    """Principal first Piola-Kirchhoff stresses (P_0, P_1, P_2) = d psi / d l_i."""
    p = _unpack(params)
    l, J, S1, S2 = _kinematics(lam)
    dI1 = 2 * J ** (-2 / 3) * (l - S1 / (3 * l))
    dI2 = J ** (-4 / 3) * (2 * l * (S1 - l ** 2) - 4 / 3 * S2 / l)
    P = p["C10"] * dI1 + 2 * (J - 1) / p["D"] * J / l
    if p["law"] != "Neo-Hookean":
        P = P + p["C01"] * dI2
    if p["law"] == "HGO":
        I1 = J ** (-2 / 3) * S1
        for ax in p["axes"]:
            E = p["kappa"] * (I1 - 3) + (1 - 3 * p["kappa"]) * (l[ax] ** 2 - 1)
            if E > 0.0:
                dE = p["kappa"] * dI1
                dE[ax] += (1 - 3 * p["kappa"]) * 2 * l[ax]
                P = P + p["k1"] * E * np.exp(p["k2"] * E ** 2) * dE
    return P


def cauchy(lam, params):
    """Principal Cauchy stresses sigma_i = P_i l_i / J."""
    l, J, _, _ = _kinematics(lam)
    return pk1(l, params) * l / J


def small_strain_moduli(params):
    """(mu, K, E, nu) of the linearisation at the reference state.
    mu = 2 (C_10 + C_01), K = 2 / D. Valid for Neo-Hookean and Mooney-Rivlin; the HGO
    fibre term is quadratic in E_i and adds a stiffness along the fibres, so it is excluded."""
    p = _unpack(params)
    if p["law"] == "HGO":
        raise ValueError("no isotropic small-strain moduli for HGO")
    mu, K = 2 * (p["C10"] + p["C01"]), 2 / p["D"]
    return mu, K, 9 * K * mu / (3 * K + mu), (3 * K - 2 * mu) / (2 * (3 * K + mu))


# --------------------------------------------------------------------------- solvers

def solve_stretches(params, conditions, guess=None, tol=1e-9):
    """Stretches (l_0, l_1, l_2) satisfying one condition per axis.

    conditions : three pairs (kind, value), kind in
        "stretch" : l_i = value,
        "cauchy"  : sigma_i = value,
        "pk1"     : P_i = value.
    The unknown stretches are solved in logarithmic form (they stay positive).
    guess : optional starting stretches (use the previous load step in a path).
    Raises RuntimeError if the solution does not satisfy the conditions to `tol`.
    """
    if len(conditions) != 3:
        raise ValueError("one condition per axis is required")
    for kind, _ in conditions:
        if kind not in ("stretch", "cauchy", "pk1"):
            raise ValueError(f"unknown condition kind {kind!r}")
    free = [i for i, (kind, _) in enumerate(conditions) if kind != "stretch"]
    lam0 = np.ones(3) if guess is None else np.array(guess, float)
    for i, (kind, value) in enumerate(conditions):
        if kind == "stretch":
            lam0[i] = value

    def build(y):
        lam = lam0.copy()
        lam[free] = np.exp(y)
        return lam

    def residual(y):
        lam = build(y)
        P = pk1(lam, params)
        sigma = P * lam / lam.prod()
        return [(sigma[i] if conditions[i][0] == "cauchy" else P[i]) - conditions[i][1]
                for i in free]

    if not free:
        return lam0
    y, _, _, msg = fsolve(residual, np.log(lam0[free]), xtol=1e-13, full_output=True)
    err = np.abs(residual(y)).max()
    # The residual is the criterion: fsolve may flag "no good progress" (ier = 5) once the
    # residual is already at round-off level, which is a converged solution here.
    if err > tol:
        raise RuntimeError(f"no solution found (residual {err:.2e}): {msg}")
    return build(y)


def _conditions(axis, kind, value):
    cond = [("cauchy", 0.0)] * 3
    cond[axis] = (kind, value)
    return cond


def uniaxial_stretch(lam_axial, axis, params, guess=None):
    """Stretch lam_axial imposed along `axis`, the two other faces traction free."""
    return solve_stretches(params, _conditions(axis, "stretch", lam_axial), guess)


def follower_pressure(p, axis, params, guess=None):
    """Follower pressure p on the face normal to `axis` (sigma_nn = -p, p > 0 compresses),
    the two other faces traction free."""
    return solve_stretches(params, _conditions(axis, "cauchy", -p), guess)


def dead_load(t, axis, params, guess=None):
    """Dead traction t (per reference area) on the face normal to `axis` (P_nn = t),
    the two other faces traction free."""
    return solve_stretches(params, _conditions(axis, "pk1", t), guess)