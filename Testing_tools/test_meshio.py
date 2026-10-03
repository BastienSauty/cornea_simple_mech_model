import gmsh
import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh as dmesh

from Simulation_Framework import read_mesh, write_mesh


def build_block(dim, recombined):
    gmsh.model.add("block")
    if dim == 2:
        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    else:
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    if recombined:   # quads / hexahedra
        for _, c in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(c, 4)
        for _, s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(s)
            gmsh.model.mesh.setRecombine(2, s)
        if dim == 3:
            for _, v in gmsh.model.getEntities(3):
                gmsh.model.mesh.setTransfiniteVolume(v)
    else:            # triangles / tetrahedra
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.addPhysicalGroup(dim, [t for _, t in gmsh.model.getEntities(dim)], 1, "domain")
    gmsh.model.addPhysicalGroup(dim - 1, [t for _, t in gmsh.model.getEntities(dim - 1)], 2, "boundary")
    gmsh.model.mesh.generate(dim)


def top_cell_centroids(dim):
    """Centroids of the top-dimension cells, in the order of _top_dim_element_tags."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.zeros((int(node_tags.max()) + 1, 3))
    xyz[node_tags.astype(int)] = coords.reshape(-1, 3)
    out = []
    for _, pg in gmsh.model.getPhysicalGroups(dim):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, pg):
            types, _, conn = gmsh.model.mesh.getElements(dim, ent)
            for t, nodes in zip(types, conn):
                n = gmsh.model.mesh.getElementProperties(t)[3]
                out.append(xyz[nodes.reshape(-1, n).astype(int)].mean(axis=1))
    return np.vstack(out)


def fibre_field(x):
    """e_1, e_2 rotated about z by an angle depending on position; e_3 = z."""
    theta = 0.5 * (x[:, 0] + 2.0 * x[:, 1])
    zero = np.zeros_like(theta)
    e_1 = np.stack([np.cos(theta), np.sin(theta), zero], axis=1)
    e_2 = np.stack([-np.sin(theta), np.cos(theta), zero], axis=1)
    e_3 = np.tile([0.0, 0.0, 1.0], (len(theta), 1))
    return e_1, e_2, e_3


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    if gmsh.isInitialized():
        gmsh.finalize()


@pytest.mark.parametrize("dim,recombined", [(2, False), (2, True), (3, False), (3, True)])
@pytest.mark.parametrize("with_lrs", [False, True])
def test_round_trip(tmp_path, gmsh_session, dim, recombined, with_lrs):
    build_block(dim, recombined)
    n_cells = len(top_cell_centroids(dim))
    kwargs = {}
    if with_lrs:
        e_1, e_2, _ = fibre_field(top_cell_centroids(dim))
        kwargs = dict(e_1=e_1, e_2=e_2)
    path = tmp_path / "block.msh"
    write_mesh(path, **kwargs)
    gmsh.clear()

    data = read_mesh(path, MPI.COMM_WORLD)
    tdim = data.domain.topology.dim
    assert tdim == dim
    assert data.domain.topology.index_map(tdim).size_local == n_cells
    assert data.has_lrs == with_lrs
    if with_lrs:
        mid = dmesh.compute_midpoints(data.domain, tdim, np.arange(n_cells))
        for got, ref in zip((data.e_1, data.e_2, data.e_3), fibre_field(mid)):
            np.testing.assert_allclose(got.x.array.reshape(-1, 3)[:n_cells], ref, atol=1e-10)


def test_checks(tmp_path, gmsh_session):
    build_block(2, False)
    e_1, e_2, e_3 = fibre_field(top_cell_centroids(2))
    path = tmp_path / "bad.msh"
    with pytest.raises(ValueError):
        write_mesh(path, e_1=e_1)                       # e_2 missing
    with pytest.raises(ValueError):
        write_mesh(path, e_1=e_1, e_2=e_2, e_3=e_3)     # e_3 in 2D
    with pytest.raises(ValueError):
        write_mesh(path, e_1=2 * e_1, e_2=e_2)          # not unit
    with pytest.raises(ValueError):
        write_mesh(path, e_1=e_1, e_2=e_1)              # not orthogonal
    with pytest.raises(ValueError):
        write_mesh(path, e_1=e_1[:-1], e_2=e_2[:-1])    # wrong length


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))