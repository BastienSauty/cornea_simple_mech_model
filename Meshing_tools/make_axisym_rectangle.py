"""Axisymmetric rectangle section [ri, re] x [0, h] in the (r, z) half-plane -> gmsh (x, y).
u: 0 central line (r = ri), 1 lateral (r = re); v: 0 bottom (z = 0), 1 top (z = h)."""
from dataclasses import dataclass

import gmsh
import numpy as np
from functools import partial

from Simulation_Framework import write_mesh
from .mesh_generation import mapped_mesh, reference_block


@dataclass
class RectangleParams:
    ri: float = 0.0       # internal radius (0: filled cylinder)
    re: float = 1.0       # external radius
    h: float = 1.0        # height
    n_r: int = 10         # elements in the radial direction
    n_z: int = 10         # elements in the axial direction


def rectangle_map(p):
    """Affine map of the unit square onto [ri, re] x [0, h]."""
    if not 0.0 <= p.ri < p.re:
        raise ValueError(f"Require 0 <= ri < re, got ri={p.ri}, re={p.re}")
    origin = np.array([p.ri, 0.0])
    scale = np.array([p.re - p.ri, p.h])
    return lambda X: origin + np.asarray(X, float) * scale


# radial direction = u, axial direction = v. The map is affine and axis-aligned, so the
# frame is constant: e_1 = [1, 0] (radial), e_2 = [0, 1] (axial). The hoop e_3 is set by read_mesh.
def frame(F):
    n = len(F)
    return np.tile([1.0, 0.0], (n, 1)), np.tile([0.0, 1.0], (n, 1))


# tags: placeholder values, to be replaced by the ones used in the simulation files
CENTRAL_LINE, LATERAL, BOTTOM, TOP = 4, 3, 2, 1

PHYSICAL = {"domain": (10, "rectangle"), "u0": (CENTRAL_LINE, "central_line"),
            "u1": (LATERAL, "lateral"), "v0": (BOTTOM, "bottom"), "v1": (TOP, "top")}


def make_rectangle(mesh_file, p=RectangleParams(), simplex=False):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("rectangle")
        lrs = mapped_mesh(rectangle_map(p),
                          partial(reference_block, (p.n_r + 1, p.n_z + 1), simplex),
                          frame, PHYSICAL)
        write_mesh(mesh_file, **lrs)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    make_rectangle("Meshing_tools/Meshes/rectangle.msh")