"""Box [0, Lx] x [0, Ly] x [0, Lz] in 3D, faces x0, x1, y0, y1, z0, z1.
Frame: e_1 = x, e_2 = y, e_3 = z. Hexahedra by default, tetrahedra with simplex=True."""
from dataclasses import dataclass

import gmsh
import numpy as np

from functools import partial
from Simulation_Framework import write_mesh
from mesh_generation import mapped_mesh, push_vector, reference_block


@dataclass
class CubeParams:
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    n_x: int = 6          # nodes along x (5 cells, as in cube_hexa.msh)
    n_y: int = 6
    n_z: int = 6


def cube_map(p):
    L = np.array([p.Lx, p.Ly, p.Lz])
    return lambda X: np.asarray(X, float) * L


# F is diagonal here, so the pushed-forward axes are the coordinate axes
frame = lambda F: (push_vector(F, [1.0, 0.0, 0.0]),
                   push_vector(F, [0.0, 1.0, 0.0]),
                   push_vector(F, [0.0, 0.0, 1.0]))

PHYSICAL = {"domain": (10, "cube"),
            "u0": (1, "x0"), "u1": (2, "x1"),
            "v0": (3, "y0"), "v1": (4, "y1"),
            "w0": (5, "z0"), "w1": (6, "z1")}


def make_cube(mesh_file, p=CubeParams(), simplex=False):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cube")
        lrs = mapped_mesh(cube_map(p), partial(reference_block, (p.n_x, p.n_y, p.n_z), simplex),
                    frame, PHYSICAL)
        write_mesh(mesh_file, **lrs)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    make_cube("Meshing_tools/Meshes/cube_hexa.msh")