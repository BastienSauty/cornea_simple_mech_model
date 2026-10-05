"""Structured hexahedral cylinder (central square + 4 outer blocks), assembled into ONE
volume. Maps phi1..phi4 as in Merlini et al., rescaled to the requested radius."""
import numpy as np
from scipy.spatial import cKDTree

from mesh_generation import discrete_mesh

L = np.log(np.sqrt(2.0))                  # outer radius sqrt(2) e^L = 2 before scaling


def outer_block(n_s, n_rho, q):
    """Nodes (n_rho+1, n_s+1, 2) of the outer block rotated by q pi/2."""
    h_s, h_rho = 2.0 / n_s, L / n_rho
    x1, x2 = np.meshgrid(np.arange(n_rho + 1), np.arange(n_s + 1), indexing="ij")
    a, b = x1 * h_rho, x2 * h_s - 1.0                                    # phi1
    alpha = (np.sqrt(2.0) * np.exp(L) - 1.0) / L
    a = L + L / (np.log(np.sqrt(2.0)) + L) * (np.log((alpha * a + 1.0) / np.sqrt(2.0)) - L)
    b = np.arctan(b)                                                      # phi2
    a = (a - L) * (np.log(np.sqrt(2.0) * np.cos(b)) / L + 1.0) + L        # phi3
    r, th = np.sqrt(2.0) * np.exp(a), b + q * np.pi / 2                   # phi4
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)


def cylinder_arrays(radius, height, n_s, n_z):
    """nodes (n, 3), hexes (m, 8) 0-based in gmsh order, faces {"bottom", "top", "lateral"}
    -> (k, 4) quads. n_s elements per side of the central square, n_rho = n_s / 2."""
    if n_s % 2:
        raise ValueError("n_s must be even (n_rho = n_s / 2)")
    n_rho = n_s // 2
    g = np.linspace(-1.0, 1.0, n_s + 1)
    square = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1)
    blocks = [square] + [outer_block(n_s, n_rho, q) for q in range(4)]

    P = np.concatenate([b.reshape(-1, 2) for b in blocks])
    tree = cKDTree(P)                                  # merge duplicate nodes on interfaces
    first = np.array([min(nb) for nb in tree.query_ball_point(P, 1e-9)])
    uniq, inv = np.unique(first, return_inverse=True)
    xy = P[uniq] * (radius / 2.0)

    quads, lateral, start = [], [], 0
    for k, b in enumerate(blocks):
        idx = inv[start:start + b[..., 0].size].reshape(b.shape[:2])
        start += idx.size
        quads.append(np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]],
                              axis=-1).reshape(-1, 4))
        if k > 0:                                      # outer arc: last radial index
            lateral.append(np.stack([idx[-1, :-1], idx[-1, 1:]], axis=-1))
    quads, lateral = np.concatenate(quads), np.concatenate(lateral)

    n2 = len(xy)
    z = np.linspace(0.0, height, n_z + 1)
    nodes = np.concatenate([np.column_stack([xy, np.full(n2, zk)]) for zk in z])
    layer = lambda k: quads + k * n2
    hexes = np.concatenate([np.hstack([layer(k), layer(k + 1)]) for k in range(n_z)])
    side = lambda k: np.hstack([lateral + k * n2, lateral[:, ::-1] + (k + 1) * n2])
    faces = {"bottom": layer(0), "top": layer(n_z),
             "lateral": np.concatenate([side(k) for k in range(n_z)])}

    # orientation check: Jacobian at the centroid of every hexahedron
    c = nodes[hexes]
    e_xi = (c[:, [1, 2, 5, 6]].sum(1) - c[:, [0, 3, 4, 7]].sum(1)) / 4
    e_eta = (c[:, [2, 3, 6, 7]].sum(1) - c[:, [0, 1, 4, 5]].sum(1)) / 4
    e_zeta = (c[:, 4:].sum(1) - c[:, :4].sum(1)) / 4
    if np.linalg.det(np.stack([e_xi, e_eta, e_zeta], axis=-1)).min() <= 0:
        raise ValueError("left-handed or degenerate hexahedron in the assembly")
    return nodes, hexes, faces


def structured_cylinder(radius, height, n_s, n_z):
    """Reference builder for mapped_mesh: hexahedral cylinder of axis z, one volume,
    faces "bottom" (z=0), "top" (z=height), "lateral"."""
    nodes, hexes, faces = cylinder_arrays(radius, height, n_s, n_z)
    return discrete_mesh(3, nodes, hexes, 5, faces, 3)