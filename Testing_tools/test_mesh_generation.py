import gmsh
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from Simulation_Framework import read_mesh, write_mesh
from Meshing_tools.mesh_generation import mapped_mesh, push_vector, push_normal

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
    lrs = mapped_mesh(phi, n_nodes, frame, physical(d), simplex)
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
        mapped_mesh(phi, (4, 4), frame, {"u0": (1, "u0")})


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))