"""Axisymmetric cornea section (r, z) -> gmsh (x, y). u: 0 central line, 1 limbus;
v: 0 posterior, 1 anterior."""
from dataclasses import dataclass

import gmsh
import numpy as np
from functools import partial
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from Simulation_Framework import write_mesh
from coons import coons_map, segment
from mesh_generation import mapped_mesh, push_vector, push_normal, reference_block


@dataclass
class CorneaParams:
    R_a: float = 7.8      # placeholder values: replace with yours
    Q_a: float = -0.1849
    R_p: float = 6.8
    Q_p: float = -0.36
    h: float = 0.53       # central thickness
    R_v: float = 5.295    # posterior radial extent (limbus)
    n_u: int = 50         # nodes
    n_v: int = 6


def f_conic(x, R, Q):
    return x**2 / R / (1 + np.sqrt(1 - (1 + Q) * x**2 / R**2))


def df_conic(x, R, Q):
    return x / np.sqrt(R**2 - (1 + Q) * x**2)


def arclength_curve(f, x_end, n_dense=2000):
    """C(u): point of (x, f(x)) at fraction u of the arc length."""
    xd = np.linspace(0.0, x_end, n_dense)
    sd = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(f(xd))))])
    x_of_s = CubicSpline(sd / sd[-1], xd)

    def C(u):
        x = x_of_s(np.clip(u, 0.0, 1.0))
        return np.stack([x, f(x)], axis=-1)
    return C


def cornea_map(p):
    """Coons map of the unit square onto the cornea section."""
    trans = p.h + f_conic(p.R_v, p.R_p, p.Q_p)
    z_ant = lambda x: trans - f_conic(x, p.R_a, p.Q_a)
    z_post = lambda x: trans - p.h - f_conic(x, p.R_p, p.Q_p)

    # limbus edge: normal to the posterior face at x = R_v, up to the anterior face
    P = np.array([p.R_v, z_post(p.R_v)])
    s = -df_conic(p.R_v, p.R_p, p.Q_p)
    normal = np.array([-s, 1.0]) / np.hypot(1.0, s)
    g = lambda t: z_ant(P[0] + t * normal[0]) - (P[1] + t * normal[1])
    x_max_ant = p.R_a / np.sqrt(1 + p.Q_a) if p.Q_a > -1 else np.inf
    t_domain = (x_max_ant - P[0]) / normal[0] if normal[0] > 0 else np.inf
    t_hi = min(5.0 * (z_ant(p.R_v) - P[1]) + 1.0, 0.999 * t_domain)
    A = P + brentq(g, 0.0, t_hi) * normal

    post = arclength_curve(z_post, p.R_v)
    ant = arclength_curve(z_ant, A[0])
    return coons_map(bottom=post, top=ant,
                     left=segment(post(0.0), ant(0.0)), right=segment(post(1.0), ant(1.0)))


# circumferential direction = u, thickness direction = v
frame = lambda F: (push_vector(F, [1.0, 0.0]), push_normal(F, [0.0, 1.0]))

PHYSICAL = {"domain": (10, "cornea"), "v0": (2, "posterior"), "v1": (1, "anterior"),
            "u1": (3, "limbus"), "u0": (4, "central_line")}


def make_cornea(mesh_file, p=CorneaParams(), simplex=False):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cornea")      
        lrs = mapped_mesh(cornea_map(p), partial(reference_block, (p.n_u, p.n_v), simplex),
                          frame, PHYSICAL)
        write_mesh(mesh_file, **lrs)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    make_cornea("Meshing_tools/Meshes/cornea.msh")