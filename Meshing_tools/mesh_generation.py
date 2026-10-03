"""Dimension-independent tools: structured (quad/hexa) or simplex (tri/tet) mesh of
the unit square/cube, deformed by a map phi, with a local frame pushed forward by phi.
The caller supplies phi, the frame recipe and the physical groups."""
import gmsh
import numpy as np

from Simulation_Framework import top_cell_centroids

AXES = "uvw"


def reference_block(n_nodes, simplex=False):
    """Mesh [0,1]^d in the current gmsh model, d = len(n_nodes), with n_nodes[i] nodes
    along axis i. Returns the domain tag and {"u0": tag, "u1": tag, "v0": ...} of faces
    (dimension d-1) found by position."""
    d = len(n_nodes)
    if d == 2:
        domain = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    elif d == 3:
        domain = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    else:
        raise ValueError("n_nodes must have 2 or 3 entries")
    gmsh.model.occ.synchronize()

    def bbox(dim, tag):
        b = np.array(gmsh.model.getBoundingBox(dim, tag))
        return b[:3], b[3:] - b[:3]

    for _, c in gmsh.model.getEntities(1):                 # edges: nodes along their axis
        _, ext = bbox(1, c)
        gmsh.model.mesh.setTransfiniteCurve(c, n_nodes[int(np.argmax(ext))])
    if not simplex:                                        # quads / hexahedra
        for _, s in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(s)
            gmsh.model.mesh.setRecombine(2, s)
        if d == 3:
            gmsh.model.mesh.setTransfiniteVolume(domain)
    gmsh.model.mesh.generate(d)

    faces = {}
    for _, f in gmsh.model.getEntities(d - 1):
        com = gmsh.model.occ.getCenterOfMass(d - 1, f)
        for a in range(d):
            if abs(com[a] - round(com[a])) < 1e-6:     # 0 or 1; the others are 0.5
                faces[f"{AXES[a]}{int(round(com[a]))}"] = f
    if len(faces) != 2 * d:
        raise RuntimeError(f"found {len(faces)} faces ({sorted(faces)}), expected {2 * d}")
        
    return domain, faces


def deform(phi, d):
    """Replace every node X (reference coordinates) by phi(X)."""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    Y = phi(xyz.reshape(-1, 3)[:, :d])
    for t, y in zip(tags, Y):
        gmsh.model.mesh.setNode(int(t), [*y, *([0.0] * (3 - d))], [])


def jacobian(phi, X, eps=1e-6):
    """F[..., i, j] = d phi_i / d X_j, central differences, shape (n, d, d).
    The points must lie at least eps inside the reference block."""
    cols = []
    for j in range(X.shape[1]):
        dX = np.zeros(X.shape[1])
        dX[j] = eps
        cols.append((phi(X + dX) - phi(X - dX)) / (2 * eps))
    return np.stack(cols, axis=-1)


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def push_vector(F, A):
    """Material direction A (reference block) -> unit vector F A."""
    return _unit(np.einsum("eij,j->ei", F, A))


def push_normal(F, N):
    """Normal direction N (reference block) -> unit vector F^-T N."""
    return _unit(np.einsum("eji,j->ei", np.linalg.inv(F), N))


def push_forward_frame(phi, frame, ref_centroids, def_centroids, tol=1e-10, max_iter=30):
    """Frame at the deformed centroids. The reference point mapped onto each deformed
    centroid is found by Newton iteration started at the reference centroid; F is taken
    there and passed to `frame`. Returns {"e_1": (n_cells, 3), ...}."""
    X = np.array(ref_centroids, float)
    for _ in range(max_iter):
        res = phi(X) - def_centroids
        if np.abs(res).max() < tol:
            break
        X -= np.linalg.solve(jacobian(phi, X), res[..., None])[..., 0]
    else:
        raise RuntimeError("inversion of phi at the deformed centroids did not converge")

    F = jacobian(phi, X)
    if np.linalg.det(F).min() <= 0:
        raise ValueError("phi does not preserve orientation (det F <= 0)")
    out = {}
    for i, e in enumerate(frame(F), start=1):
        out[f"e_{i}"] = np.hstack([e, np.zeros((len(e), 3 - e.shape[1]))])
    return out


def mapped_mesh(phi, n_nodes, frame, physical, simplex=False):
    """Mesh the reference block, add physical groups, deform by phi, return the LRS dict.
    physical = {"domain": (tag, name), "u0": (tag, name), ...}; faces not listed are
    not written."""
    if "domain" not in physical:
        raise ValueError('physical must contain a "domain" entry')
    d = len(n_nodes)
    domain, faces = reference_block(n_nodes, simplex)
    
    for key, (tag, name) in physical.items():
        if key == "domain":
            gmsh.model.addPhysicalGroup(d, [domain], tag, name)
        else:
            gmsh.model.addPhysicalGroup(d - 1, [faces[key]], tag, name)
    ref = top_cell_centroids(d)[:, :d]
    deform(phi, d)
    return push_forward_frame(phi, frame, ref, top_cell_centroids(d)[:, :d])