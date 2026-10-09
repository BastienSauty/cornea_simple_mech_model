"""Shared fixtures of the verification tests.

Nothing here imports dolfinx or gmsh at module level: the analytical self-tests
(test_analytical.py) must run on a machine without them. The mesh factories import them
when first called and skip the requesting test if they are missing.

Meshes are built in a temporary directory with the Meshing_tools machinery, so no mesh file
is committed for these tests. Each factory returns (mesh_file, tags) where `tags` maps a
face name to its physical tag.
"""
import json
from functools import partial

import numpy as np
import pytest

CUBE_TAGS = {"x0": 1, "x1": 2, "y0": 3, "y1": 4, "z0": 5, "z1": 6, "domain": 10}

# Rectangle [r_min, r_max] x [0, length] of the (r, z) half-plane (axisymmetric frameworks).
# For a solid cylinder r_min = 0 and "r_min" is the symmetry axis.
RECT_TAGS = {"r_min": 1, "r_max": 2, "z0": 3, "z1": 4, "domain": 10}
_RECT_PHYSICAL = {"domain": (10, "domain"), "u0": (1, "r_min"), "u1": (2, "r_max"),
                  "v0": (3, "z0"), "v1": (4, "z1")}


@pytest.fixture
def write_params(tmp_path):
    """write_params(dict) -> path of a JSON material file, as read by from_json."""
    def _write(params, name="params.json"):
        path = tmp_path / name
        path.write_text(json.dumps(params, indent=2))
        return str(path)
    return _write


@pytest.fixture(scope="session")
def cube_mesh(tmp_path_factory):
    """cube_mesh(simplex=False, n=3, size=(1, 1, 1)) -> (mesh_file, tags).

    Box [0, Lx] x [0, Ly] x [0, Lz] with n cells per side, hexahedra (simplex=False) or
    tetrahedra. Faces x0, x1, y0, y1, z0, z1 and the LRS e_1, e_2, e_3 = x, y, z.
    """
    pytest.importorskip("dolfinx")
    pytest.importorskip("gmsh")
    from make_cube import CubeParams, make_cube          # Meshing_tools, see pytest.ini

    cache = {}

    def build(simplex=False, n=3, size=(1.0, 1.0, 1.0)):
        key = (bool(simplex), int(n), tuple(size))
        if key not in cache:
            folder = tmp_path_factory.mktemp("cube")
            path = str(folder / f"cube_{'tet' if simplex else 'hex'}_{n}.msh")
            make_cube(path, CubeParams(*size, n + 1, n + 1, n + 1), simplex)   # n_x.. are nodes
            cache[key] = path
        return cache[key], dict(CUBE_TAGS)

    return build


@pytest.fixture(scope="session")
def rectangle_mesh(tmp_path_factory):
    """rectangle_mesh(r_min, r_max, length, n_r, n_z, simplex=False) -> (mesh_file, tags).

    2D mesh of [r_min, r_max] x [0, length] in (r, z), quadrilaterals or triangles, with
    n_r x n_z cells. Faces r_min, r_max, z0, z1. LRS: e_1 = radial, e_2 = axial (e_3 is the
    hoop direction, set by read_mesh).
    """
    pytest.importorskip("dolfinx")
    pytest.importorskip("gmsh")
    import gmsh
    from mesh_generation import mapped_mesh, push_normal, push_vector, reference_block
    from Simulation_Framework import write_mesh

    cache = {}

    def build(r_min, r_max, length, n_r, n_z, simplex=False):
        key = (r_min, r_max, length, n_r, n_z, bool(simplex))
        if key not in cache:
            folder = tmp_path_factory.mktemp("rectangle")
            path = str(folder / f"rect_{'tri' if simplex else 'quad'}_{n_r}x{n_z}.msh")
            scale, shift = np.array([r_max - r_min, length]), np.array([r_min, 0.0])
            phi = lambda X: np.asarray(X, float) * scale + shift
            frame = lambda F: (push_vector(F, [1.0, 0.0]), push_normal(F, [0.0, 1.0]))
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            try:
                gmsh.model.add("rectangle")
                lrs = mapped_mesh(phi, partial(reference_block, (n_r + 1, n_z + 1), simplex),
                                  frame, _RECT_PHYSICAL)
                write_mesh(path, **lrs)
            finally:
                gmsh.finalize()
            cache[key] = path
        return cache[key], dict(RECT_TAGS)

    return build