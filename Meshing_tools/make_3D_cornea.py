"""3D cornea from a tetrahedral cylinder (X1, X2, X3) -> (x, y, z), apex up.
X3 = 0 posterior, X3 = h_t anterior. Biconic surfaces; xi1 extremities of the posterior
surface at z = 0; limbus normal to the posterior surface."""
from dataclasses import dataclass
from functools import partial

import gmsh
import numpy as np

from Simulation_Framework import write_mesh
from mesh_generation import reference_cylinder, mapped_mesh, push_vector, push_normal
from structured_cylinder import structured_cylinder

@dataclass
class Cornea3DParams:
    R_H: float = 5.93
    R_V: float = 5.25
    h_t: float = 0.5
    R1_p: float = 5.5
    R1_a: float = 6.5
    R2_p: float = 4.8
    R2_a: float = 5.8
    Q1_a: float = -0.61
    Q1_p: float = -0.72
    Q2_a: float = -0.61
    Q2_p: float = -0.72
    size: float = 0.17      # uniform element size of the reference cylinder

    structured: bool = True  # hexahedra (Merlini assembly) or tetrahedra
    n_s: int = 4            # elements per side of the central square (n_rho = n_s / 2)
    n_z: int = 2             # elements through the thickness


def biconic(x, y, R1, R2, Q1, Q2):
    """Sag f(x, y) and its gradient (f_x, f_y)."""
    S = 1 - (1 + Q1) * x**2 / R1**2 - (1 + Q2) * y**2 / R2**2
    root = np.sqrt(S)
    D = 1 + root
    N = x**2 / R1 + y**2 / R2
    k = N / (D**2 * root)
    return (N / D,
            2 * x / (R1 * D) + k * (1 + Q1) * x / R1**2,
            2 * y / (R2 * D) + k * (1 + Q2) * y / R2**2)

def cornea_map(p):
    post = lambda x, y: biconic(x, y, p.R1_p, p.R2_p, p.Q1_p, p.Q2_p)
    ant = lambda x, y: biconic(x, y, p.R1_a, p.R2_a, p.Q1_a, p.Q2_a)

    def limbus_offset(angle, n_iter=20):
        """Horizontal displacement from the anterior outline point at `angle` (on the ellipse
        R_H, R_V) to the posterior outline, along the inward normal to the anterior surface.
        The wall meets the posterior surface after a length T. Independent of the shift C0."""
        xe, ye = p.R_H * np.cos(angle), p.R_V * np.sin(angle)
        fa, fax, fay = ant(xe, ye)
        N = np.sqrt(fax**2 + fay**2 + 1.0)
        nx, ny, nz = -fax / N, -fay / N, -1.0 / N
        T = np.full_like(angle, p.h_t)
        for _ in range(n_iter):                  # g(T) = 0: wall point on the posterior surface
            fp, fpx, fpy = post(xe + T * nx, ye + T * ny)
            g = p.h_t - fa + T * nz + fp
            T = T - g / (nz + fpx * nx + fpy * ny)
        return np.stack([T * nx, T * ny], axis=-1)

    # lowest point of the posterior outline (xi1 extremity) at z = 0
    off0 = limbus_offset(np.array([0.0]))
    C0 = post(p.R_H + off0[0, 0], 0.0)[0]
    z_post = lambda x, y: C0 - post(x, y)[0]
    z_ant = lambda x, y: C0 + p.h_t - ant(x, y)[0]

    def phi(X):
        X = np.asarray(X, float)
        s = X[..., :2] / p.R_H                   # unit disk
        r2 = (s**2).sum(-1)
        x, y = p.R_H * s[..., 0], p.R_V * s[..., 1]
        t = (X[..., 2] / p.h_t)[..., None]
        angle = np.arctan2(s[..., 1], s[..., 0])
        P_xy = np.stack([x, y], axis=-1) + r2[..., None] * limbus_offset(angle)
        P = np.concatenate([P_xy, z_post(P_xy[..., 0], P_xy[..., 1])[..., None]], axis=-1)
        A = np.stack([x, y, z_ant(x, y)], axis=-1)
        return (1 - t) * P + t * A
    return phi

def frame(F):
    """e_1 = F e_X1 and e_3 = F^-T e_X3 exact (orthogonal by construction), e_2 = e_3 x e_1."""
    e_1 = push_vector(F, [1.0, 0.0, 0.0])
    e_3 = push_normal(F, [0.0, 0.0, 1.0])
    e_2 = np.cross(e_3, e_1)
    cos = np.abs((e_2 * push_vector(F, [0.0, 1.0, 0.0])).sum(1))
    print(f"max angle between e_2 and the push-forward of X2: "
          f"{np.degrees(np.arccos(np.clip(cos, 0, 1))).max():.2f} deg")
    return e_1, e_2, e_3


PHYSICAL = {"domain": (10, "cornea"), "bottom": (2, "posterior"),
            "top": (1, "anterior"), "lateral": (3, "limbus")}


def make_cornea_3d(mesh_file, p=Cornea3DParams()):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cornea_3d")
        
        if p.structured:
            reference = partial(structured_cylinder, p.R_H, p.h_t, p.n_s, p.n_z)
        else:
            reference = partial(reference_cylinder, p.R_H, p.h_t, p.size)
        lrs = mapped_mesh(cornea_map(p), reference, frame, PHYSICAL)
        write_mesh(mesh_file, **lrs)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    make_cornea_3d("Meshing_tools/Meshes/cornea_3D_hex.msh")