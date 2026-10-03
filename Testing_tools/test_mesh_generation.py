import gmsh
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh
from functools import partial

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # /workspace, if the test is in Testing_tools/
sys.path[:0] = [str(ROOT / "Simulation_Framework"), str(ROOT / "Meshing_tools")]

from Simulation_Framework import read_mesh, write_mesh
from Meshing_tools.mesh_generation import mapped_mesh, push_vector, push_normal, reference_cylinder, reference_block
from Meshing_tools.structured_cylinder import cylinder_arrays, structured_cylinder

SHIFT = 0.3


def phi(X):
    """Last coordinate += SHIFT * u^2. F = I + 2 SHIFT u (e_last x e_u)."""
    Y = np.array(X, float)
    Y[..., -1] += SHIFT * X[..., 0] ** 2
    return Y


def frame(F):
    d = F.shape[1]
    if d == 2:
        return push_vector(F, [1.0, 0.0]), push_normal(F, [0.0, 1.0])
    e_1 = push_vector(F, [1.0, 0.0, 0.0])
    e_3 = push_normal(F, [0.0, 0.0, 1.0])
    e_2 = np.cross(e_3, e_1)
    return e_1, e_2, e_3


def expected(x, d):
    a = 2 * SHIFT * x
    one, zero = np.ones_like(a), np.zeros_like(a)
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    if d == 2:
        return unit(np.stack([one, a, zero], 1)), unit(np.stack([-a, one, zero], 1))
    return (unit(np.stack([one, zero, a], 1)), np.stack([zero, one, zero], 1),
            unit(np.stack([-a, zero, one], 1)))


def physical(d):
    names = ["u0", "u1", "v0", "v1", "w0", "w1"][: 2 * d]
    out = {"domain": (10, "domain")}
    out.update({n: (i + 1, n) for i, n in enumerate(names)})
    return out


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    if gmsh.isInitialized():
        gmsh.finalize()


@pytest.mark.parametrize("n_nodes", [(6, 4), (4, 3, 3)])
@pytest.mark.parametrize("simplex", [False, True])
def test_mapped_mesh(tmp_path, gmsh_session, n_nodes, simplex):
    d = len(n_nodes)
    lrs = mapped_mesh(phi, partial(reference_block, n_nodes, simplex), frame, physical(d))
    path = tmp_path / "mapped.msh"
    write_mesh(path, **lrs)
    gmsh.clear()

    data = read_mesh(path, MPI.COMM_WORLD)
    domain, tdim = data.domain, data.domain.topology.dim
    assert tdim == d
    n_cells = domain.topology.index_map(tdim).size_local
    if not simplex:
        assert n_cells == np.prod(np.array(n_nodes) - 1)

    # geometry: u in [0, 1], last coordinate in [0, 1 + SHIFT]
    x = domain.geometry.x
    assert x[:, 0].min() == pytest.approx(0.0) and x[:, 0].max() == pytest.approx(1.0)
    assert x[:, d - 1].min() == pytest.approx(0.0)
    assert x[:, d - 1].max() == pytest.approx(1.0 + SHIFT)

    # physical groups: every face tag is present
    assert set(np.unique(data.facet_tag.values)) == set(range(1, 2 * d + 1))

    # local frame against the analytic values at the cell centroids
    assert data.has_lrs
    mid = dmesh.compute_midpoints(domain, tdim, np.arange(n_cells))
    got = (data.e_1, data.e_2, data.e_3)[:d + (d == 3)]
    for g, ref in zip(got, expected(mid[:, 0], d)):
        np.testing.assert_allclose(g.x.array.reshape(-1, 3)[:n_cells], ref, atol=1e-7)


def test_missing_domain(gmsh_session):
    with pytest.raises(ValueError):
        mapped_mesh(phi, partial(reference_block, (4, 4)), frame, {"u0": (1, "u0")})


def phi_cyl(X):
    Y = np.array(X, float)
    Y[..., 2] += 0.3 * (X[..., 0] ** 2 + X[..., 1] ** 2)
    return Y


def frame_cyl(F):
    e_1 = push_vector(F, [1.0, 0.0, 0.0])
    e_3 = push_normal(F, [0.0, 0.0, 1.0])
    return e_1, np.cross(e_3, e_1), e_3

@pytest.mark.parametrize("kind", ["tet", "hexa"])
def test_cylinder(tmp_path, gmsh_session, kind):
    reference = {"tet": partial(reference_cylinder, 1.0, 0.5, 0.25),
                 "hexa": partial(structured_cylinder, 1.0, 0.5, 8, 2)}[kind]
    physical = {"domain": (10, "cyl"), "bottom": (1, "bottom"), "top": (2, "top"),
                "lateral": (3, "lateral")}
    lrs = mapped_mesh(phi_cyl, reference, frame_cyl, physical)
    path = tmp_path / "cyl.msh"
    write_mesh(path, **lrs)
    gmsh.clear()

    data = read_mesh(path, MPI.COMM_WORLD)
    domain = data.domain
    assert domain.topology.dim == 3 and data.has_lrs
    assert set(np.unique(data.facet_tag.values)) == {1, 2, 3}
    n_cells = domain.topology.index_map(3).size_local
    mid = dmesh.compute_midpoints(domain, 3, np.arange(n_cells))
    ax, ay = 0.6 * mid[:, 0], 0.6 * mid[:, 1]
    one, zero = np.ones_like(ax), np.zeros_like(ax)
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    e_1 = unit(np.stack([one, zero, ax], 1))
    e_3 = unit(np.stack([-ax, -ay, one], 1))
    for g, ref in zip((data.e_1, data.e_2, data.e_3), (e_1, np.cross(e_3, e_1), e_3)):
        np.testing.assert_allclose(g.x.array.reshape(-1, 3)[:n_cells], ref, atol=1e-7)


def test_cylinder_arrays():
    n_s, n_z = 16, 3
    nodes, hexes, faces = cylinder_arrays(1.0, 0.5, n_s, n_z)
    n_rho = n_s // 2
    assert len(hexes) == n_z * (n_s**2 + 4 * n_s * n_rho)
    assert len(nodes) == (n_z + 1) * ((n_s + 1) ** 2 + 4 * n_rho * n_s)
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    assert r.max() == pytest.approx(1.0)
    assert np.allclose(r[np.unique(faces["lateral"])], 1.0)
    assert np.allclose(nodes[np.unique(faces["bottom"]), 2], 0.0)
    assert np.allclose(nodes[np.unique(faces["top"]), 2], 0.5)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))