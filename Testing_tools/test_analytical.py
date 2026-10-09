"""Checks of analytical.py itself (numpy, scipy, sympy: no dolfinx needed).

The finite element verification tests compare the framework with analytical.py. These
tests make sure the reference is right first, so that a failure of a finite element test
points at the framework and not at its reference.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import analytical as an

sp = pytest.importorskip("sympy")


# Parameter sets: the three reference laws, plus HGO with dispersion and with both
# families on the same axis (exercises the kappa terms and the double counting).
CASES = {
    "neo-hookean": an.reference_params("Neo-Hookean"),
    "mooney-rivlin": an.reference_params("Mooney-Rivlin"),
    "hgo-cube": an.reference_params("HGO", "cube"),
    "hgo-axi": an.reference_params("HGO", "axi"),
    "hgo-kappa": {**an.reference_params("HGO"), "kappa": 0.15, "a4": [0, 0, 1], "a6": [0, 0, 1]},
}
ISOTROPIC = ["neo-hookean", "mooney-rivlin"]


# ------------------------------------------------------------------ independent oracle

def sympy_pk1(params):
    """P_i = d psi / d F_ii by symbolic differentiation, psi written with the matrix
    definitions (C, traces, a.C.a) and none of the hand-derived principal-stretch formulas."""
    l = sp.symbols("l0:3", positive=True)
    F = sp.diag(*l)
    C = F.T * F
    J = F.det()
    Cb = J ** sp.Rational(-2, 3) * C
    I1 = Cb.trace()
    I2 = (I1 ** 2 - (Cb * Cb).trace()) / 2
    psi = params["C_10"] * (I1 - 3) + (J - 1) ** 2 / params["D"]
    if params["sedf_type"] != "Neo-Hookean":
        psi += params["C_01"] * (I2 - 3)
    if params["sedf_type"] == "HGO":
        k1, k2, kap = params["k1"], params["k2"], params["kappa"]
        for a in (params["a4"], params["a6"]):
            a = sp.Matrix(a) / sp.sqrt(sum(c ** 2 for c in a))
            E = kap * (I1 - 3) + (1 - 3 * kap) * ((a.T * C * a)[0] - 1)
            psi += k1 / (2 * k2) * sp.Piecewise((sp.exp(k2 * E ** 2) - 1, E > 0), (0, True))
    grads = sp.lambdify(l, [sp.diff(psi, li) for li in l], modules="numpy")
    return lambda lam: np.array(grads(*lam), float)


@pytest.mark.parametrize("case", CASES)
def test_pk1_matches_sympy(case):
    params = CASES[case]
    oracle = sympy_pk1(params)
    points = np.random.default_rng(0).uniform(0.7, 1.4, (25, 3))   # extension and compression
    for lam in points:
        assert_allclose(an.pk1(lam, params), oracle(lam), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_pk1_is_gradient_of_energy(case):
    params, eps = CASES[case], 1e-6
    for lam in np.random.default_rng(1).uniform(0.75, 1.35, (10, 3)):
        fd = [(an.energy(lam + eps * e, params) - an.energy(lam - eps * e, params)) / (2 * eps)
              for e in np.eye(3)]
        assert_allclose(an.pk1(lam, params), fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("case", CASES)
def test_reference_state_is_stress_free(case):
    assert_allclose(an.pk1([1, 1, 1], CASES[case]), 0.0, atol=1e-14)
    assert an.energy([1, 1, 1], CASES[case]) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("case", ISOTROPIC)
def test_isotropic_laws_are_permutation_covariant(case):
    lam, perm = np.array([1.3, 0.9, 1.1]), [1, 2, 0]
    assert_allclose(an.pk1(lam[perm], CASES[case]), an.pk1(lam, CASES[case])[perm], rtol=1e-13)


# ------------------------------------------------------------------ HGO fibres

def test_hgo_fibres_are_off_in_compression():
    """kappa = 0 and fibres along 0 and 1: E_i = l_i^2 - 1 < 0, the law reduces to Mooney-Rivlin."""
    hgo, mr = an.reference_params("HGO"), an.reference_params("Mooney-Rivlin")
    lam = [0.9, 0.8, 1.2]
    assert_allclose(an.pk1(lam, hgo), an.pk1(lam, mr), rtol=1e-13)
    assert an.energy(lam, hgo) == pytest.approx(an.energy(lam, mr), rel=1e-13)


@pytest.mark.parametrize("frame, active", [("cube", (0,)), ("axi", (0, 2))])
def test_hgo_fibre_increment_is_along_the_fibre(frame, active):
    """kappa = 0: the fibre term adds k1 E exp(k2 E^2) 2 l_i to P_i on the fibre axes only."""
    hgo, mr = an.reference_params("HGO", frame), an.reference_params("Mooney-Rivlin")
    lam = np.array([1.2, 1.0, 1.1]) if frame == "axi" else np.array([1.2, 0.9, 1.0])
    expected = np.zeros(3)
    for ax in active:
        E = lam[ax] ** 2 - 1
        expected[ax] = hgo["k1"] * E * np.exp(hgo["k2"] * E ** 2) * 2 * lam[ax]
    assert_allclose(an.pk1(lam, hgo) - an.pk1(lam, mr), expected, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ limits

@pytest.mark.parametrize("case", ISOTROPIC)
def test_small_strain_limit(case):
    params, eps = CASES[case], 1e-5
    mu, K, E, nu = an.small_strain_moduli(params)
    lam = an.uniaxial_stretch(1 + eps, 2, params)
    sigma = an.cauchy(lam, params)
    assert sigma[2] / eps == pytest.approx(E, rel=1e-3)
    assert (1 - lam[0]) / eps == pytest.approx(nu, rel=1e-3)
    assert_allclose(sigma[:2], 0.0, atol=1e-10)


@pytest.mark.parametrize("case", ISOTROPIC)
@pytest.mark.parametrize("stretch", [0.8, 1.3, 1.8])
def test_incompressible_limit_uniaxial(case, stretch):
    """D -> 0: classical result sigma = 2 (l^2 - 1/l)(C_10 + C_01 / l)."""
    params = {**CASES[case], "D": 1e-3}
    lam = an.uniaxial_stretch(stretch, 2, params)
    c01 = 0.0 if case == "neo-hookean" else params["C_01"]
    sigma_inc = 2 * (stretch ** 2 - 1 / stretch) * (params["C_10"] + c01 / stretch)
    assert an.cauchy(lam, params)[2] == pytest.approx(sigma_inc, rel=2e-2)
    assert lam.prod() == pytest.approx(1.0, abs=2e-2)


# ------------------------------------------------------------------ solvers

@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("stretch", [0.8, 1.0, 1.3])
def test_uniaxial_stretch_is_traction_free_laterally(case, stretch):
    params = CASES[case]
    axis = 2
    lam = an.uniaxial_stretch(stretch, axis, params)
    sigma = an.cauchy(lam, params)
    assert lam[axis] == stretch
    assert_allclose(np.delete(sigma, axis), 0.0, atol=1e-9)


@pytest.mark.parametrize("case", ["neo-hookean", "mooney-rivlin", "hgo-cube"])
def test_lateral_stretches_are_equal_when_the_frame_is_symmetric(case):
    """Isotropic laws, and HGO with the same fibre family along x and y."""
    lam = an.uniaxial_stretch(1.25, 2, CASES[case])
    assert lam[0] == pytest.approx(lam[1], rel=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("p", [-0.5, -0.1, 0.1, 0.5])
def test_follower_pressure_state(case, p):
    sigma = an.cauchy(an.follower_pressure(p, 2, CASES[case]), CASES[case])
    assert_allclose(sigma, [0.0, 0.0, -p], atol=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("t", [-0.3, 0.3, 0.8])
def test_dead_load_state(case, t):
    P = an.pk1(an.dead_load(t, 2, CASES[case]), CASES[case])
    assert_allclose(P, [0.0, 0.0, t], atol=1e-9)


def test_follower_load_follows_the_current_area():
    """With the same p, the reference traction P_nn and the Cauchy stress differ by the area
    ratio: P_zz = sigma_zz J / l_z. A dead load of that value gives the same state."""
    params = CASES["mooney-rivlin"]
    lam = an.follower_pressure(0.4, 2, params)
    P_zz = an.pk1(lam, params)[2]
    assert_allclose(an.dead_load(P_zz, 2, params), lam, rtol=1e-9)


def test_fully_prescribed_stretches_are_returned_unchanged():
    lam = an.solve_stretches(an.reference_params("HGO"), [("stretch", v) for v in (1.1, 0.9, 1.2)])
    assert_allclose(lam, [1.1, 0.9, 1.2])


# ------------------------------------------------------------------ input validation

def test_invalid_inputs():
    with pytest.raises(ValueError):
        an.reference_params("Ogden")
    with pytest.raises(ValueError):
        an.pk1([1, -1, 1], an.reference_params("Mooney-Rivlin"))
    oblique = {**an.reference_params("HGO"), "a4": [1, 1, 0]}
    with pytest.raises(ValueError):
        an.pk1([1, 1, 1], oblique)
    with pytest.raises(ValueError):
        an.small_strain_moduli(an.reference_params("HGO"))
    with pytest.raises(ValueError):
        an.solve_stretches(an.reference_params("Neo-Hookean"), [("stretch", 1.0)] * 2)